"""
Tennis Intelligence System - Search Tools
=========================================

Search tools for web information retrieval and intelligent result summarization.
Integrates with the existing tavily_search.py utility and adds LLM-based processing.
"""

import time
import json
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import the existing search utility
import sys
from pathlib import Path

# Add utils directory to path to import tavily_search
utils_path = Path(__file__).parent.parent / "utils"
sys.path.append(str(utils_path))

try:
    from tavily_search import tavily_search
    from src.config.settings import TennisConfig, get_agent_prompt
except ImportError:
    # Fallback import paths
    from utils.tavily_search import tavily_search
    from config.settings import TennisConfig, get_agent_prompt


@tool
def optimize_search_query(user_query: str, context: str = "") -> Dict[str, Any]:
    """
    Optimize a user query for more effective web search using tennis domain expertise.
    
    This tool takes a user's natural language question and transforms it into 
    optimized search terms that will yield better results from web search engines.
    Uses current date information programmatically rather than relying on LLM knowledge.
    
    Args:
        user_query: The original user question/query
        context: Additional context about the query intent
        
    Returns:
        Dictionary containing optimized search query and reasoning
    """
    config = TennisConfig()
    
    try:
        # Get current date information programmatically
        from datetime import datetime
        now = datetime.now()
        current_year = now.year
        current_date = now.strftime("%Y-%m-%d")
        
        llm = ChatOpenAI(model=config.default_model, temperature=0.1)
        
        optimization_prompt = f"""Optimize this tennis query for web search.

CURRENT: {current_date} ({current_year})
QUERY: "{user_query}"

Transform into effective search terms:
- Add {current_year} for current/recent queries  
- Use official tennis terms (ATP, WTA, rankings)
- Include relevant keywords
- Remove conversational words

Examples:
"Who's the best player?" → "ATP WTA world rankings {current_year} current top tennis players"
"Recent tournament results" → "latest tennis tournament {current_year} results winner"

Optimized search query:"""

        response = llm.invoke([{"role": "user", "content": optimization_prompt}])
        optimized_query = response.content.strip()
        
        # Remove quotes if the LLM added them
        if optimized_query.startswith('"') and optimized_query.endswith('"'):
            optimized_query = optimized_query[1:-1]
        
        return {
            "success": True,
            "original_query": user_query,
            "optimized_query": optimized_query,
            "context_used": context,
            "current_date": current_date,
            "current_year": current_year,
            "optimization_applied": True,
            "timestamp": time.time()
        }
        
    except Exception as e:
        # Fallback to original query if optimization fails
        return {
            "success": False,
            "original_query": user_query,
            "optimized_query": user_query,  # Fallback to original
            "context_used": context,
            "optimization_applied": False,
            "error": str(e),
            "timestamp": time.time()
        }


@tool
def tavily_search_tool(query: str, max_results: int = 10) -> Dict[str, Any]:
    """
    Perform web search using Tavily API for tennis-related information.
    
    This tool searches the web for current tennis information, news,
    rankings, and other real-time data not available in the database.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary containing raw search results and metadata
    """
    config = TennisConfig()
    start_time = time.time()
    
    try:
        # Use the existing tavily_search function
        raw_results = tavily_search(query)
        
        execution_time = time.time() - start_time
        
        # Structure the results
        search_results = {
            "success": True,
            "query_used": query,
            "raw_results": raw_results,
            "execution_time": execution_time,
            "timestamp": time.time(),
            "result_count": len(raw_results.get("results", [])),
            "sources": [result.get("url", "") for result in raw_results.get("results", [])],
            "error": None
        }
        
        return search_results
        
    except Exception as e:
        return {
            "success": False,
            "query_used": query,
            "raw_results": {},
            "execution_time": time.time() - start_time,
            "timestamp": time.time(),
            "result_count": 0,
            "sources": [],
            "error": f"Search failed: {str(e)}"
        }


@tool
def interpret_search_results(search_results: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    """
    Convert raw search results to natural language interpretation using a single efficient LLM call.
    
    Args:
        search_results: Raw search results from tavily_search_tool
        user_query: Original user question for context
        
    Returns:
        Dictionary with natural language interpretation
    """
    config = TennisConfig()
    
    if not search_results.get("success", False):
        return {
            "success": False,
            "interpretation": f"Web search failed: {search_results.get('error', 'Unknown error')}",
            "confidence": 0.0
        }
    
    raw_results = search_results.get("raw_results", {})
    results_list = raw_results.get("results", [])
    
    if not results_list:
        return {
            "success": True,
            "interpretation": "No relevant information found in web search results.",
            "confidence": 0.3
        }
    
    try:
        llm = ChatOpenAI(model=config.default_model, temperature=0.2)
        
        # Prepare search results for interpretation (limit to top 3 for efficiency)
        top_results = results_list[:3]
        search_summary = ""
        
        for i, result in enumerate(top_results, 1):
            title = result.get("title", "No title")
            content = result.get("content", "No content")[:300]  # Truncate long content
            source = result.get("url", "").split("//")[-1].split("/")[0] if result.get("url") else "Unknown"
            
            search_summary += f"{i}. {title}\n   Source: {source}\n   Content: {content}...\n\n"
        
        # Create efficient interpretation prompt
        interpretation_prompt = f"""Convert these web search results into clear, natural language for a tennis question.

USER QUESTION: "{user_query}"

WEB SEARCH RESULTS:
{search_summary}

Provide a clear, conversational summary of the most relevant and current information found. Focus on facts, rankings, recent developments, or current status. Be specific with names, numbers, and dates when available.

Example responses:
- "According to recent rankings, Player X is currently ranked #Y..."
- "The latest tournament results show that..."
- "Current information indicates that..."

Natural language interpretation:"""

        response = llm.invoke([
            SystemMessage(content="You are a tennis information interpreter. Convert web search results into clear, current information summaries."),
            HumanMessage(content=interpretation_prompt)
        ])
        
        interpretation = response.content.strip()
        
        return {
            "success": True,
            "interpretation": interpretation,
            "source_count": len(results_list),
            "has_data": True,
            "confidence": 0.8
        }
        
    except Exception as e:
        return {
            "success": False,
            "interpretation": f"Failed to interpret search results: {str(e)}",
            "confidence": 0.0
        }


@tool
def filter_tennis_content(search_results: Dict[str, Any], filter_criteria: str = "tennis_relevance") -> Dict[str, Any]:
    """
    Filter search results to focus on tennis-relevant content.
    
    Args:
        search_results: Search results to filter
        filter_criteria: Type of filtering to apply
        
    Returns:
        Filtered search results
    """
    if not search_results.get("success", False):
        return search_results
    
    raw_results = search_results.get("raw_results", {})
    results_list = raw_results.get("results", [])
    
    # Tennis-related keywords for filtering
    tennis_keywords = [
        'tennis', 'atp', 'wta', 'grand slam', 'wimbledon', 'us open', 
        'french open', 'australian open', 'roland garros', 'serve', 'volley',
        'match', 'tournament', 'ranking', 'player', 'court', 'clay', 'grass', 
        'hard court', 'surface', 'set', 'game', 'point', 'racket', 'ace'
    ]
    
    filtered_results = []
    
    for result in results_list:
        title = result.get("title", "").lower()
        content = result.get("content", "").lower()
        url = result.get("url", "").lower()
        
        # Calculate tennis relevance score
        relevance_score = 0
        for keyword in tennis_keywords:
            if keyword in title:
                relevance_score += 3  # Title matches are weighted higher
            if keyword in content:
                relevance_score += 1
            if keyword in url:
                relevance_score += 2
        
        # Apply filtering threshold
        if filter_criteria == "tennis_relevance" and relevance_score >= 2:
            result["tennis_relevance_score"] = relevance_score
            filtered_results.append(result)
        elif filter_criteria == "strict_tennis" and relevance_score >= 5:
            result["tennis_relevance_score"] = relevance_score
            filtered_results.append(result)
        elif filter_criteria == "any_mention" and relevance_score >= 1:
            result["tennis_relevance_score"] = relevance_score
            filtered_results.append(result)
    
    # Update search results with filtered content
    filtered_search_results = search_results.copy()
    filtered_search_results["raw_results"]["results"] = filtered_results
    filtered_search_results["result_count"] = len(filtered_results)
    filtered_search_results["filter_applied"] = filter_criteria
    filtered_search_results["original_count"] = len(results_list)
    
    return filtered_search_results


@tool
def extract_recent_updates(search_results: Dict[str, Any], days_threshold: int = 7) -> Dict[str, Any]:
    """
    Extract the most recent updates from search results.
    
    Args:
        search_results: Search results to analyze
        days_threshold: How many days back to consider "recent"
        
    Returns:
        Dictionary with recent updates and analysis
    """
    if not search_results.get("success", False):
        return {
            "success": False,
            "recent_updates": [],
            "recency_analysis": "No search results to analyze"
        }
    
    raw_results = search_results.get("raw_results", {})
    results_list = raw_results.get("results", [])
    
    # Keywords that indicate recent/current information
    recent_keywords = [
        'today', 'yesterday', 'this week', 'latest', 'breaking', 'current',
        'recent', 'now', 'live', 'update', 'just', '2024', '2025',
        'currently', 'ongoing', 'developing'
    ]
    
    recent_updates = []
    
    for result in results_list:
        title = result.get("title", "").lower()
        content = result.get("content", "").lower()
        
        # Calculate recency score
        recency_score = 0
        for keyword in recent_keywords:
            if keyword in title:
                recency_score += 2
            if keyword in content:
                recency_score += 1
        
        if recency_score > 0:
            result_copy = result.copy()
            result_copy["recency_score"] = recency_score
            result_copy["recency_indicators"] = [
                keyword for keyword in recent_keywords 
                if keyword in title or keyword in content
            ]
            recent_updates.append(result_copy)
    
    # Sort by recency score
    recent_updates.sort(key=lambda x: x.get("recency_score", 0), reverse=True)
    
    return {
        "success": True,
        "recent_updates": recent_updates[:5],  # Top 5 most recent
        "total_recent_count": len(recent_updates),
        "recency_analysis": f"Found {len(recent_updates)} results with recency indicators",
        "top_recency_indicators": list(set([
            indicator for update in recent_updates[:3] 
            for indicator in update.get("recency_indicators", [])
        ]))
    }


def _create_basic_summary(results_list: List[Dict], query: str, error: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a basic summary when LLM processing fails.
    
    Args:
        results_list: List of search results
        query: Original search query
        error: Optional error message
        
    Returns:
        Basic summary dictionary
    """
    summarized_results = []
    
    for result in results_list:
        # Extract basic information
        source_url = result.get("url", "")
        source_domain = source_url.split("//")[-1].split("/")[0] if source_url else "unknown"
        
        # Simple credibility assessment based on domain
        credible_domains = [
            'atptour.com', 'wtatennis.com', 'itftennis.com', 'usopen.org',
            'wimbledon.com', 'rolandgarros.com', 'ausopen.com',
            'espn.com', 'bbc.com', 'cnn.com', 'reuters.com'
        ]
        
        credibility = "high" if any(domain in source_domain for domain in credible_domains) else "medium"
        
        # Basic relevance scoring
        title = result.get("title", "").lower()
        content = result.get("content", "").lower()
        
        relevance = 0.8 if "tennis" in title else 0.6 if "tennis" in content else 0.4
        
        summarized_results.append({
            "source": source_domain,
            "url": source_url,
            "summary": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
            "relevance_score": relevance,
            "recency": "unknown",
            "credibility": credibility
        })
    
    return {
        "success": True,
        "summarized_results": summarized_results,
        "key_findings": f"Found {len(results_list)} results for query: {query}",
        "current_information": "Processing completed with basic summarization",
        "information_gaps": "Advanced LLM processing unavailable",
        "source_quality_notes": f"Processed {len(results_list)} sources with basic assessment",
        "confidence_assessment": 0.6,
        "processing_note": f"Basic processing used{': ' + error if error else ''}"
    } 