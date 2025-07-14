"""
Tennis Intelligence System - Text Processing Tools
==================================================

Text analysis and processing tools for entity extraction, sentiment analysis,
and content relevance scoring to support the tennis intelligence agents.
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

try:
    from src.config.settings import TennisConfig
except ImportError:
    from config.settings import TennisConfig


@tool
def extract_key_entities(text: str, entity_types: List[str] = None) -> Dict[str, Any]:
    """
    Extract tennis-related entities from text using pattern matching and LLM.
    
    Args:
        text: Text to analyze for entities
        entity_types: Types of entities to extract (players, tournaments, surfaces, etc.)
        
    Returns:
        Dictionary with extracted entities and metadata
    """
    if entity_types is None:
        entity_types = ["players", "tournaments", "surfaces", "years", "rankings"]
    
    entities = {
        "players": [],
        "tournaments": [],
        "surfaces": [],
        "years": [],
        "rankings": [],
        "other": []
    }
    
    text_lower = text.lower()
    
    # Extract years (1950-2030)
    year_pattern = r'\b(19[5-9][0-9]|20[0-3][0-9])\b'
    years = re.findall(year_pattern, text)
    entities["years"] = list(set(years))
    
    # Extract surfaces
    surface_keywords = ["clay", "hard", "grass", "carpet", "indoor", "outdoor"]
    for surface in surface_keywords:
        if surface in text_lower:
            entities["surfaces"].append(surface.title())
    
    # Extract tournament names (pattern-based)
    tournament_patterns = [
        r'\b(Australian Open|French Open|US Open|Wimbledon)\b',
        r'\b(Roland Garros|All England Club)\b',
        r'\b(\w+\s+Open)\b',
        r'\b(\w+\s+Masters)\b',
        r'\b(\w+\s+Cup)\b',
        r'\b(ATP\s+\d+|WTA\s+\d+)\b'
    ]
    
    for pattern in tournament_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["tournaments"].extend(matches)
    
    # Extract ranking mentions
    ranking_pattern = r'\b(?:rank|ranked|ranking)\s*[#]?(\d+)\b'
    rankings = re.findall(ranking_pattern, text_lower)
    entities["rankings"] = [f"#{rank}" for rank in rankings]
    
    # Player name extraction (more complex - using common patterns)
    entities["players"] = _extract_player_names(text)
    
    # Remove duplicates and empty entries
    for entity_type in entities:
        entities[entity_type] = list(set([e for e in entities[entity_type] if e.strip()]))
    
    return {
        "success": True,
        "entities": entities,
        "entity_count": sum(len(v) for v in entities.values()),
        "text_length": len(text),
        "extraction_confidence": _calculate_extraction_confidence(entities, text)
    }


@tool
def analyze_sentiment(text: str, focus_aspect: str = "general") -> Dict[str, Any]:
    """
    Analyze sentiment of tennis-related text.
    
    Args:
        text: Text to analyze
        focus_aspect: Specific aspect to focus on (player, match, performance, etc.)
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # Simple keyword-based sentiment analysis
    positive_keywords = [
        "win", "victory", "champion", "success", "excellent", "outstanding",
        "brilliant", "amazing", "dominant", "crushing", "spectacular",
        "perfect", "ace", "winner", "breakthrough", "comeback"
    ]
    
    negative_keywords = [
        "loss", "defeat", "struggle", "poor", "terrible", "awful",
        "disappointing", "injured", "withdrawal", "retire", "upset",
        "error", "fault", "miss", "failed", "crashed"
    ]
    
    neutral_keywords = [
        "match", "game", "set", "tournament", "play", "compete",
        "serve", "return", "point", "court", "surface"
    ]
    
    text_lower = text.lower()
    
    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    negative_count = sum(1 for word in negative_keywords if word in text_lower)
    neutral_count = sum(1 for word in neutral_keywords if word in text_lower)
    
    total_sentiment_words = positive_count + negative_count + neutral_count
    
    if total_sentiment_words == 0:
        sentiment_score = 0.0  # Neutral
        sentiment_label = "neutral"
    else:
        # Calculate weighted sentiment score (-1 to 1)
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        if sentiment_score > 0.2:
            sentiment_label = "positive"
        elif sentiment_score < -0.2:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
    
    # Confidence based on number of sentiment indicators
    confidence = min(total_sentiment_words / 10, 1.0)  # Capped at 1.0
    
    return {
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "confidence": confidence,
        "positive_indicators": positive_count,
        "negative_indicators": negative_count,
        "neutral_indicators": neutral_count,
        "total_sentiment_words": total_sentiment_words,
        "text_length": len(text),
        "focus_aspect": focus_aspect
    }


@tool
def calculate_relevance_score(text: str, query_context: str, domain: str = "tennis") -> Dict[str, Any]:
    """
    Calculate how relevant a piece of text is to a given query context.
    
    Args:
        text: Text to evaluate
        query_context: Original query or context to compare against
        domain: Domain for relevance scoring (default: tennis)
        
    Returns:
        Dictionary with relevance scoring results
    """
    text_lower = text.lower()
    query_lower = query_context.lower()
    
    # Split into words for analysis
    text_words = set(re.findall(r'\b\w+\b', text_lower))
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Calculate word overlap
    word_overlap = len(text_words.intersection(query_words))
    total_unique_words = len(text_words.union(query_words))
    
    if total_unique_words == 0:
        word_relevance = 0.0
    else:
        word_relevance = word_overlap / len(query_words) if query_words else 0.0
    
    # Domain-specific relevance (tennis keywords)
    tennis_keywords = [
        'tennis', 'atp', 'wta', 'grand slam', 'tournament', 'match',
        'player', 'ranking', 'serve', 'volley', 'court', 'surface',
        'clay', 'grass', 'hard', 'set', 'game', 'point', 'ace',
        'winner', 'forehand', 'backhand', 'net', 'baseline'
    ]
    
    domain_relevance = 0.0
    if domain == "tennis":
        tennis_mentions = sum(1 for keyword in tennis_keywords if keyword in text_lower)
        domain_relevance = min(tennis_mentions / 10, 1.0)  # Capped at 1.0
    
    # Entity overlap (check for shared named entities)
    entities_text = _extract_simple_entities(text)
    entities_query = _extract_simple_entities(query_context)
    
    entity_overlap = len(entities_text.intersection(entities_query))
    entity_relevance = entity_overlap / max(len(entities_query), 1)
    
    # Weighted final score
    final_score = (
        word_relevance * 0.4 +
        domain_relevance * 0.3 +
        entity_relevance * 0.3
    )
    
    # Classify relevance level
    if final_score >= 0.7:
        relevance_level = "high"
    elif final_score >= 0.4:
        relevance_level = "medium"
    else:
        relevance_level = "low"
    
    return {
        "relevance_score": final_score,
        "relevance_level": relevance_level,
        "word_overlap": word_overlap,
        "word_relevance": word_relevance,
        "domain_relevance": domain_relevance,
        "entity_relevance": entity_relevance,
        "entity_overlap_count": entity_overlap,
        "text_length": len(text),
        "query_length": len(query_context),
        "analysis_confidence": min((len(text) + len(query_context)) / 1000, 1.0)
    }


def _extract_player_names(text: str) -> List[str]:
    """
    Extract potential player names from text using pattern matching.
    
    Args:
        text: Text to extract names from
        
    Returns:
        List of potential player names
    """
    # Common tennis player name patterns
    # Format: FirstName LastName (often with initials)
    name_patterns = [
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]*\.?)\b',  # John Smith or John S.
        r'\b([A-Z]\.\s*[A-Z][a-z]+)\b',         # J. Smith
        r'\b([A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+)\b'  # John A. Smith
    ]
    
    potential_names = []
    
    for pattern in name_patterns:
        matches = re.findall(pattern, text)
        potential_names.extend(matches)
    
    # Filter out common false positives
    false_positives = [
        "New York", "Los Angeles", "Grand Slam", "US Open", "French Open",
        "Australian Open", "First Set", "Second Set", "Third Set",
        "Match Point", "Game Point", "Set Point"
    ]
    
    filtered_names = [
        name for name in potential_names 
        if name not in false_positives and len(name.split()) >= 2
    ]
    
    return list(set(filtered_names))


def _extract_simple_entities(text: str) -> set:
    """
    Extract simple entities (capitalized words/phrases) from text.
    
    Args:
        text: Text to extract entities from
        
    Returns:
        Set of potential entities
    """
    # Extract capitalized words that might be entities
    entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    entities = set(re.findall(entity_pattern, text))
    
    # Filter out common non-entities
    common_words = {
        "The", "This", "That", "These", "Those", "A", "An", "In", "On", 
        "At", "By", "For", "With", "And", "Or", "But", "So", "If", "When"
    }
    
    return {entity for entity in entities if entity not in common_words}


def _calculate_extraction_confidence(entities: Dict[str, List], text: str) -> float:
    """
    Calculate confidence score for entity extraction based on various factors.
    
    Args:
        entities: Extracted entities dictionary
        text: Original text
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    total_entities = sum(len(v) for v in entities.values())
    text_length = len(text)
    
    # Base confidence on entity density and text length
    if text_length == 0:
        return 0.0
    
    entity_density = total_entities / (text_length / 100)  # Entities per 100 characters
    
    # Normalize to 0-1 range
    confidence = min(entity_density / 2, 1.0)  # Assuming 2 entities per 100 chars is high confidence
    
    # Boost confidence if we found entities of multiple types
    entity_type_count = sum(1 for v in entities.values() if v)
    type_bonus = min(entity_type_count / 5, 0.3)  # Up to 0.3 bonus for entity diversity
    
    return min(confidence + type_bonus, 1.0) 