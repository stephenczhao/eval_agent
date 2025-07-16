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
import os

# Import the existing search utility
import sys
from pathlib import Path

# Add utils directory to path to import tavily_search
utils_path = Path(__file__).parent.parent / "utils"
sys.path.append(str(utils_path))

from ..utils.tavily_search import tavily_search
from ..config.settings import TennisConfig
from ..config.optimized_prompts import get_optimized_prompt


def _debug_print(message: str) -> None:
    """Print message only if debug mode is enabled via environment variable."""
    if os.environ.get('TENNIS_DEBUG', 'False').lower() == 'true':
        print(message)


@tool
def optimize_search_query(user_query: str, context: str = "") -> Dict[str, Any]:
    """
    Optimize a user query for more effective web search using tennis domain expertise.
    
    Args:
        user_query: The original user question/query
        context: Additional context about the query intent
        
    Returns:
        Dictionary containing optimized search query and reasoning
    """
    config = TennisConfig()
    

    llm = ChatOpenAI(
        model=config.default_model,
        temperature=0.1,
        max_tokens=200,
        api_key=config.openai_api_key
    )
    
    # Use optimized search prompt with datetime context from get_optimized_prompt
    search_prompt = get_optimized_prompt('search')
    optimization_prompt = f"""Transform this tennis query into effective web search terms:

USER QUERY: "{user_query}"

SEARCH OPTIMIZATION EXAMPLES:
- "who's the best player right now?" → "current ATP rankings world number 1 tennis"
- "last tournament winner" → "latest tennis tournament winner ATP WTA recent"
- "when did he win his last tournament?" → "recent tournament wins tennis championships latest"
- "current tennis rankings" → "current ATP WTA rankings latest tennis"

STRATEGY:
- Use "current", "latest", "recent" for temporal context (NOT specific dates)
- Include "ATP" or "WTA" for official sources  
- Add "tennis" as core keyword
- Be specific about what you're looking for
- Avoid future dates - use relative terms only

Generate effective search terms (max 15 words):"""

    response = llm.invoke([
        SystemMessage(content=search_prompt),
        HumanMessage(content=optimization_prompt)
    ])
    
    optimized_query = response.content.strip()
    
    # Ensure 15-word limit for better search effectiveness
    words = optimized_query.split()
    if len(words) > 15:
        optimized_query = ' '.join(words[:15])
        _debug_print(f"   ⚠️ Truncated query to 15 words: {optimized_query}")
    
    return {
        "success": True,
        "optimized_query": optimized_query,
        "original_query": user_query,
        "optimization_applied": True,
        "reasoning": "LLM-optimized search query for tennis domain"
    }

@tool
def tavily_search_tool(query: str, max_results: int = 10) -> Dict[str, Any]:
    """
    Perform web search using Tavily API for tennis-related information.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary containing raw search results and metadata
    """
    config = TennisConfig()
    start_time = time.time()
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
            "interpretation": f"Search failed: {search_results.get('error', 'Unknown error')}",
            "confidence": 0.0,
            "source_count": 0,
            "has_data": False
        }
    
    raw_results = search_results.get("raw_results", {})
    results_list = raw_results.get("results", [])
    
    if not results_list:
        return {
            "success": False,
            "interpretation": "No search results found for the query",
            "confidence": 0.0,
            "source_count": 0,
            "has_data": False
        }

    llm = ChatOpenAI(
        model=config.default_model,
        temperature=0.2,
        max_tokens=300,
        api_key=config.openai_api_key
    )
    
    # Create concise results summary for LLM processing
    results_summary = []
    for i, result in enumerate(results_list[:5]):  # Limit to first 5 results
        title = result.get("title", "No title")
        content = result.get("content", "No content")[:300]  # Limit content length
        results_summary.append(f"Result {i+1}: {title}\nContent: {content}")
    
    results_text = "\n\n".join(results_summary)
    
    # Use optimized prompt for interpretation with datetime context
    search_prompt = get_optimized_prompt('search')
    interpretation_prompt = f"""USER QUESTION: "{user_query}"\n\nSEARCH RESULTS:\n{results_text}\n\nProvide a factual summary in 2-3 sentences focusing on direct answer, current tennis information, and key facts/figures:"""

    response = llm.invoke([
        SystemMessage(content=search_prompt),
        HumanMessage(content=interpretation_prompt)
    ])
    
    interpretation = response.content.strip()
    
    return {
        "success": True,
        "interpretation": interpretation,
        "confidence": 0.8,
        "source_count": len(results_list),
        "has_data": True
    }

@tool
def online_search(user_query: str, context: str = "") -> Dict[str, Any]:
    """
    Complete online search analysis combining query optimization, web search, and result interpretation.
    
    This tool handles the entire search pipeline:
    1. Optimizes the query for better web search results
    2. Performs web search using Tavily API
    3. Interprets and summarizes the results
    
    Args:
        user_query: The user's tennis question or query
        context: Additional context about the query intent
        
    Returns:
        Dictionary containing complete search analysis with interpretation
    """
    config = TennisConfig()
    start_time = time.time()
    
    _debug_print(f"🌐 Starting complete online search for: '{user_query}'")
    
    try:
        # Step 1: Query Optimization
        _debug_print("✨ Step 1: Optimizing search query...")
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.1,
            max_tokens=200,
            api_key=config.openai_api_key
        )
        
        search_prompt = get_optimized_prompt('search')
        optimization_prompt = f"""USER QUERY: "{user_query}"\nCONTEXT: {context}\n\nCreate search terms for current rankings, official sources (ATP, WTA, ESPN), and recent content. 

IMPORTANT: Use relative terms like "current", "latest", "recent" - NOT specific dates. Limit your response to 20 words maximum. Return only optimized search keywords:"""
        
        try:
            response = llm.invoke([
                SystemMessage(content=search_prompt),
                HumanMessage(content=optimization_prompt)
            ])
            optimized_query = response.content.strip()
            
            # Ensure 20-word limit
            words = optimized_query.split()
            if len(words) > 20:
                optimized_query = ' '.join(words[:20])
                _debug_print(f"   ⚠️ Truncated query to 20 words: {optimized_query}")
            
            _debug_print(f"   ✅ Optimized query: {optimized_query}")
        except Exception as e:
            optimized_query = f"tennis latest current {user_query}"
            
            # Ensure fallback doesn't exceed 20 words
            words = optimized_query.split()
            if len(words) > 20:
                optimized_query = ' '.join(words[:20])
            
            _debug_print(f"   ⚠️ Using fallback optimization: {optimized_query}")
        
        # Step 2: Web Search
        _debug_print("🔍 Step 2: Performing web search...")
        try:
            raw_results = tavily_search(optimized_query)
            results_list = raw_results.get("results", [])
            _debug_print(f"   ✅ Found {len(results_list)} search results")
        except Exception as e:
            _debug_print(f"   ❌ Search failed: {str(e)}")
            return {
                "success": False,
                "error": f"Web search failed: {str(e)}",
                "interpretation": "Unable to perform web search",
                "optimized_query": optimized_query,
                "execution_time": time.time() - start_time,
                "tools_called": ['optimize_search_query', 'tavily_search_tool', 'interpret_search_results']
            }
        
        if not results_list:
            return {
                "success": False,
                "error": "No search results found",
                "interpretation": "No relevant tennis information found for this query",
                "optimized_query": optimized_query,
                "execution_time": time.time() - start_time,
                "tools_called": ['optimize_search_query', 'tavily_search_tool', 'interpret_search_results']
            }
        
        # Step 3: Result Interpretation
        _debug_print("🎾 Step 3: Interpreting search results...")
        try:
            # Create concise results summary for LLM processing
            results_summary = []
            for i, result in enumerate(results_list[:5]):  # Limit to first 5 results
                title = result.get("title", "No title")
                content = result.get("content", "No content")[:300]  # Limit content length
                results_summary.append(f"Result {i+1}: {title}\nContent: {content}")
            
            results_text = "\n\n".join(results_summary)
            
            interpretation_prompt = f"""USER QUESTION: "{user_query}"\n\nSEARCH RESULTS:\n{results_text}\n\nProvide a factual summary in 2-3 sentences focusing on direct answer, current tennis information, and key facts/figures:"""

            response = llm.invoke([
                SystemMessage(content=search_prompt),
                HumanMessage(content=interpretation_prompt)
            ])
            
            interpretation = response.content.strip()
            _debug_print(f"   ✅ Interpretation complete: {interpretation[:100]}{'...' if len(interpretation) > 100 else ''}")
            
        except Exception as e:
            interpretation = f"Found {len(results_list)} search results for tennis query: {user_query}. Results include information from various tennis sources but detailed interpretation is unavailable."
            _debug_print(f"   ⚠️ Using fallback interpretation: {str(e)}")
        
        execution_time = time.time() - start_time
        _debug_print(f"🎯 Online search finished in {execution_time:.2f}s")
        
        return {
            "success": True,
            "interpretation": interpretation,
            "optimized_query": optimized_query,
            "result_count": len(results_list),
            "sources": [result.get("url", "") for result in results_list[:5]],
            "confidence": 0.8 if len(results_list) > 0 else 0.3,
            "execution_time": execution_time,
            "tools_called": ['optimize_search_query', 'tavily_search_tool', 'interpret_search_results']
        }
        
    except Exception as e:
        error_msg = f"Complete online search failed: {str(e)}"
        _debug_print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "interpretation": f"Online search encountered an error: {str(e)}",
            "execution_time": time.time() - start_time,
            "tools_called": ['optimize_search_query', 'tavily_search_tool', 'interpret_search_results']
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