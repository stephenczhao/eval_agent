"""
Tennis Intelligence System - Text Processing Tools
==================================================

Text analysis and processing tools for entity extraction to support the tennis intelligence agents.
"""

import re
from typing import Dict, Any, List
from langchain_core.tools import tool

try:
    from src.config.settings import TennisConfig
except ImportError:
    from config.settings import TennisConfig


@tool
def extract_key_entities(text: str, entity_types: List[str] = None) -> Dict[str, Any]:
    """
    Extract tennis-related entities from text using pattern matching.
    
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
    
    # Player name extraction
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


def _extract_player_names(text: str) -> List[str]:
    """Extract potential player names from text using pattern matching."""
    players = []
    
    # Common tennis player name patterns
    name_patterns = [
        r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # First Last
        r'\b([A-Z]\. [A-Z][a-z]+)\b',      # R. Federer
        r'\b([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)\b',  # Rafael N. Nadal
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, text)
        players.extend(matches)
    
    # Filter out common non-player words
    stop_words = {"Open", "Cup", "Tournament", "Championship", "Masters", "Final", "Semi", "Quarter"}
    players = [p for p in players if not any(word in p for word in stop_words)]
    
    return players


def _calculate_extraction_confidence(entities: Dict[str, List], text: str) -> float:
    """Calculate confidence score for entity extraction."""
    total_entities = sum(len(v) for v in entities.values())
    text_length = len(text.split())
    
    if text_length == 0:
        return 0.0
    
    # Base confidence on entity density and text length
    density_score = min(total_entities / text_length, 1.0)
    length_bonus = min(text_length / 100, 0.5)  # Bonus for longer text
    
    confidence = (density_score * 0.7) + (length_bonus * 0.3)
    return min(confidence, 1.0) 