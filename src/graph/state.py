"""
Tennis Intelligence System - State Management
=============================================

Defines the state structure for the LangGraph workflow that manages
the flow of information between different agents in the system.
"""

from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from datetime import datetime


class TennisIntelligenceState(TypedDict):
    """
    Main state object for the Tennis Intelligence System workflow.
    
    This state is passed between all agents and contains all the information
    needed to process a user query from start to finish.
    """
    
    # === User Interaction ===
    user_query: str
    """The original user question or request"""
    
    conversation_memory: List[Dict[str, Any]]
    """Previous queries and responses for context"""
    
    # === Query Analysis ===
    query_intent: Optional[str]
    """Classified intent: 'statistical', 'current_events', 'general', 'mixed'"""
    
    tennis_entities: List[str]
    """Extracted tennis entities (players, tournaments, etc.)"""
    
    time_context: Optional[str]
    """Temporal context: 'historical', 'current', 'recent', 'mixed'"""
    
    # === Routing Decisions ===
    routing_decision: Dict[str, Any]
    """
    Routing decisions made by orchestrator:
    {
        "sql_needed": bool,
        "search_needed": bool, 
        "reasoning": str,
        "query_complexity": str,  # 'simple', 'moderate', 'complex'
        "estimated_confidence": float
    }
    """
    
    # === Data Collection ===
    sql_results: Optional[Dict[str, Any]]
    """
    Results from SQL database queries:
    {
        "success": bool,
        "columns": List[str],
        "rows": List[List[Any]],
        "row_count": int,
        "formatted": str,
        "error": Optional[str],
        "query_used": str,
        "execution_time": float
    }
    """
    
    search_results: Optional[Dict[str, Any]]
    """
    Results from web search:
    {
        "raw_results": dict,  # Original Tavily results
        "summarized_results": List[dict],  # Processed summaries
        "sources": List[str],
        "search_queries": List[str],
        "relevance_scores": List[float],
        "execution_time": float
    }
    """
    
    # === Analysis and Processing ===
    sql_analysis: Optional[str]
    """SQL agent's interpretation of database results"""
    
    search_analysis: Optional[str]
    """Search agent's summary of web results"""
    
    data_conflicts: List[Dict[str, Any]]
    """Any conflicts found between SQL and search data"""
    
    # === Final Response ===
    synthesized_response: str
    """Final comprehensive response combining all sources"""
    
    response_structure: Dict[str, Any]
    """
    Metadata about the response structure:
    {
        "sections": List[str],  # Response sections
        "primary_source": str,  # 'sql', 'search', 'both'
        "confidence_breakdown": Dict[str, float],
        "completeness_score": float
    }
    """
    
    # === Quality and Metadata ===
    confidence_score: float
    """Overall confidence in the response (0.0 to 1.0)"""
    
    sources_used: List[str]
    """List of data sources used in the response"""
    
    response_quality_metrics: Dict[str, float]
    """
    Quality metrics for evaluation:
    {
        "relevance": float,
        "completeness": float, 
        "accuracy": float,
        "timeliness": float
    }
    """
    
    # === Execution Metadata ===
    execution_time: float
    """Total execution time in seconds"""
    
    start_time: datetime
    """When the query processing started"""
    
    agent_execution_times: Dict[str, float]
    """Execution time for each agent"""
    
    errors_encountered: List[Dict[str, Any]]
    """Any errors or warnings during processing"""
    
    # === Debug and Monitoring ===
    debug_info: Dict[str, Any]
    """Debug information for development and monitoring"""
    
    trace_id: Optional[str]
    """JudgEval trace ID for evaluation tracking"""


@dataclass
class ConversationMemoryEntry:
    """Structure for conversation memory entries."""
    timestamp: datetime
    user_query: str
    system_response: str
    sources_used: List[str]
    confidence_score: float
    execution_time: float


@dataclass  
class EntityExtraction:
    """Structure for extracted tennis entities."""
    entity_type: str  # 'player', 'tournament', 'surface', 'year', etc.
    entity_value: str
    confidence: float
    context: str  # Where in the query it was found


def create_initial_state(user_query: str, conversation_memory: Optional[List[Dict[str, Any]]] = None) -> TennisIntelligenceState:
    """
    Create an initial state object for a new query.
    
    Args:
        user_query: The user's question or request
        conversation_memory: Previous conversation context
        
    Returns:
        Initialized TennisIntelligenceState
    """
    return TennisIntelligenceState(
        # User interaction
        user_query=user_query,
        conversation_memory=conversation_memory or [],
        
        # Query analysis
        query_intent=None,
        tennis_entities=[],
        time_context=None,
        
        # Routing
        routing_decision={},
        
        # Data collection
        sql_results=None,
        search_results=None,
        
        # Analysis
        sql_analysis=None,
        search_analysis=None,
        data_conflicts=[],
        
        # Response
        synthesized_response="",
        response_structure={},
        
        # Quality
        confidence_score=0.0,
        sources_used=[],
        response_quality_metrics={},
        
        # Execution metadata
        execution_time=0.0,
        start_time=datetime.now(),
        agent_execution_times={},
        errors_encountered=[],
        
        # Debug
        debug_info={},
        trace_id=None
    ) 