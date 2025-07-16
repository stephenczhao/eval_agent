"""
Tennis Intelligence System - Orchestrator Agent
===============================================

The Orchestrator Agent analyzes tennis queries and determines the optimal routing strategy.
It breaks down tasks and routes them to SQL (database) or Search (web) agents based on the
query requirements.
"""

import json
import re
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from src.config.settings import TennisConfig
    from src.config.optimized_prompts import get_optimized_prompt
    from src.tools.text_processing_tools import extract_key_entities
    from src.utils.memory_manager import MemoryManager
except ImportError:
    # Fallback for different import contexts
    from ..config.settings import TennisConfig
    from ..config.optimized_prompts import get_optimized_prompt
    from ..tools.text_processing_tools import extract_key_entities
    from ..utils.memory_manager import MemoryManager


class OrchestratorAgent:
    """
    Orchestrator agent that analyzes tennis queries and makes routing decisions.
    
    Responsibilities:
    - Break down user query into required tasks
    - Determine if tasks need SQL database query or web search
    - Route to appropriate agents based on data requirements
    """
    
    def __init__(self, config: TennisConfig, memory_manager: MemoryManager):
        """
        Initialize the Orchestrator Agent.
        
        Args:
            config: Tennis system configuration
            memory_manager: Memory management instance for conversation context
        """
        self.config = config
        self.memory_manager = memory_manager
        self.llm = ChatOpenAI(
            model=config.default_model,
            temperature=config.temperature,
            api_key=config.openai_api_key
        )
        self.system_prompt = get_optimized_prompt('orchestrator')
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response, handling cases where JSON is embedded in other text.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed JSON dictionary or fallback structure
        """
        if not response_text or not response_text.strip():
            return self._get_fallback_structure("Empty response from LLM")
        
        # Try to parse the response directly as JSON first
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Look for JSON patterns in the response
        # Pattern 1: JSON between ```json and ```
        json_pattern1 = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern1, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Pattern 2: JSON between ``` and ```
        json_pattern2 = r'```\s*(\{.*?\})\s*```'
        match = re.search(json_pattern2, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Pattern 3: First { to last } in the response
        start_brace = response_text.find('{')
        end_brace = response_text.rfind('}')
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            try:
                json_text = response_text[start_brace:end_brace + 1]
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass
        
        # If all JSON extraction attempts fail, return fallback
        return self._get_fallback_structure(
            f"Could not extract valid JSON from response: {response_text[:200]}..."
        )
    
    def _get_fallback_structure(self, error_message: str) -> Dict[str, Any]:
        """
        Create a fallback analysis structure when JSON parsing fails.
        
        Args:
            error_message: Description of the parsing error
            
        Returns:
            Valid analysis structure with error information
        """
        return {
            "query_analysis": {
                "intent": "unknown",
                "tennis_entities": [],
                "time_context": "mixed",
                "complexity": "moderate",
                "is_follow_up": False,
                "tennis_relevance": "medium"
            },
            "routing_decision": {
                "sql_needed": False,
                "search_needed": True,
                "reasoning": f"JSON parsing failed, defaulting to search: {error_message}",
                "priority": "search_first",
                "estimated_confidence": 0.3
            },
            "context_analysis": {
                "memory_relevant": False,
                "follow_up_potential": True,
                "conversation_context_used": False
            },
            "parsing_error": error_message,
            "fallback_used": True
        }
    
    def _validate_analysis_structure(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix the analysis structure to ensure required fields exist.
        
        Args:
            analysis: Analysis result from LLM
            
        Returns:
            Validated and corrected analysis structure
        """
        # Ensure required top-level keys exist
        required_keys = ["query_analysis", "routing_decision", "context_analysis"]
        for key in required_keys:
            if key not in analysis:
                analysis[key] = {}
        
        # Validate query_analysis with minimal fallbacks
        query_defaults = {
            "intent": "unknown",
            "tennis_entities": [],
            "time_context": "mixed",
            "complexity": "moderate",
            "is_follow_up": False,
            "tennis_relevance": "medium"
        }
        for key, default_value in query_defaults.items():
            if key not in analysis["query_analysis"]:
                analysis["query_analysis"][key] = default_value
        
        # Validate routing_decision with neutral fallbacks that don't bias toward any source
        routing_defaults = {
            "sql_needed": False,
            "search_needed": False,
            "reasoning": "LLM analysis required for routing decision",
            "priority": "parallel",
            "estimated_confidence": 0.5
        }
        for key, default_value in routing_defaults.items():
            if key not in analysis["routing_decision"]:
                analysis["routing_decision"][key] = default_value
        
        # If neither source was selected, default to letting LLM choose
        if not analysis["routing_decision"]["sql_needed"] and not analysis["routing_decision"]["search_needed"]:
            analysis["routing_decision"]["search_needed"] = True
            analysis["routing_decision"]["reasoning"] = "Defaulted to search when no source was selected"
        
        # Validate context_analysis
        context_defaults = {
            "memory_relevant": False,
            "follow_up_potential": True,
            "conversation_context_used": False
        }
        for key, default_value in context_defaults.items():
            if key not in analysis["context_analysis"]:
                analysis["context_analysis"][key] = default_value
        
        return analysis
    
    def analyze_and_route(
        self, 
        user_query: str, 
        session_id: str,
        conversation_memory: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the user query and determine routing strategy.
        
        Args:
            user_query: The user's question or request
            session_id: Session identifier for memory management
            conversation_memory: Previous conversation context
            
        Returns:
            Dictionary with query analysis and routing decisions
        """
        start_time = time.time()
        
        try:
            # Extract tennis entities from the query
            entity_extraction = extract_key_entities.invoke({"text": user_query})
            tennis_entities = []
            for entity_type, entities in entity_extraction.get("entities", {}).items():
                tennis_entities.extend(entities)
            
            # Get relevant conversation context
            relevant_context = self.memory_manager.get_conversation_history(
                session_id, max_pairs=3
            )
            
            # Create analysis prompt with context
            analysis_prompt = self._create_analysis_prompt(
                user_query, tennis_entities, relevant_context
            )
            
            # Get LLM analysis
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Extract and parse JSON from the response
            analysis_result = self._extract_json_from_response(response.content)
            
            # Validate and fix the structure
            analysis_result = self._validate_analysis_structure(analysis_result)
            
            # Add metadata
            analysis_result.update({
                "execution_time": time.time() - start_time,
                "extracted_entities": tennis_entities,
                "entity_confidence": entity_extraction.get("extraction_confidence", 0.0),
                "context_used": len(relevant_context),
                "session_id": session_id,
                "timestamp": time.time()
            })
            
            return analysis_result
            
        except Exception as e:
            # Enhanced error fallback with more specific error information
            return {
                "success": False,
                "error": f"Orchestrator analysis failed: {str(e)}",
                "query_analysis": {
                    "intent": "unknown",
                    "tennis_entities": [],
                    "time_context": "mixed",
                    "complexity": "moderate",
                    "is_follow_up": False,
                    "tennis_relevance": "medium"
                },
                "routing_decision": {
                    "sql_needed": False,
                    "search_needed": True,
                    "reasoning": f"Exception occurred during analysis: {str(e)}",
                    "priority": "search_first",
                    "estimated_confidence": 0.3
                },
                "context_analysis": {
                    "memory_relevant": False,
                    "follow_up_potential": True,
                    "conversation_context_used": False
                },
                "execution_time": time.time() - start_time,
                "exception_type": type(e).__name__,
                "fallback_used": True
            }
    
    def _create_analysis_prompt(
        self,
        user_query: str,
        tennis_entities: List[str],
        relevant_context: List[Dict[str, Any]]
    ) -> str:
        """
        Create the analysis prompt with relevant context.
        
        Args:
            user_query: User's question
            tennis_entities: Extracted tennis entities
            relevant_context: Relevant conversation history
            
        Returns:
            Formatted prompt string
        """
        # Get current datetime context
        current_datetime = datetime.now()
        current_date_str = current_datetime.strftime("%Y-%m-%d")
        current_year = current_datetime.year
        current_month = current_datetime.strftime("%B %Y")
        
        # Build conversation context if available
        context_summary = ""
        if relevant_context:
            # Get the most recent relevant conversation
            most_recent = relevant_context[0]  # Most relevant is first
            prev_query = most_recent.user_query
            prev_response_snippet = most_recent.system_response[:300]  # First 300 chars
            
            context_summary = f"""
RECENT CONVERSATION CONTEXT:
Previous Question: "{prev_query}"
Previous Answer Summary: "{prev_response_snippet}..."

IMPORTANT: If the current query seems incomplete or refers to previous context (like "who's the second?", "what about X?", "and the next one?"), interpret it in relation to the previous conversation."""
        
        entities_summary = ""
        if tennis_entities:
            entities_summary = f"\nEXTRACTED ENTITIES: {', '.join(tennis_entities)}\n"
        
        prompt = f"""
        ANALYZE THIS TENNIS QUERY FOR DATA SOURCE SELECTION:
        
        CURRENT CONTEXT:
        - Today's date: {current_date_str}
        - Current year: {current_year}
        - Current period: {current_month}
        
        AVAILABLE DATA SOURCES:
        - SQL Database: Tennis matches from 2023-{current_year}, comprehensive historical data, player stats, tournament results
        - Web Search: Current rankings, recent news, live tournaments, breaking tennis information
        {context_summary}
        
        USER QUERY: "{user_query}"
        {entities_summary}
        
        ANALYSIS TASK:
        Consider what information the user is seeking and determine which data source(s) would best answer their query. You have access to both historical database records and current web information.
        
        DECISION FACTORS TO CONSIDER:
        - If this appears to be a follow-up question, consider the previous conversation context
        - Whether the query needs historical data, current information, or both
        - The temporal scope of the information being requested
        - Whether statistical analysis or real-time updates would better serve the user
        
        Use your judgment to determine the optimal data source strategy.
        
        Provide your analysis in the exact JSON format specified in your system prompt.
        Make sure to return ONLY valid JSON without any additional text or explanations.
        """
        
        return prompt