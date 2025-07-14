"""
Tennis Intelligence System - Clarification Agent
===============================================

The Clarification Agent handles unclear queries, off-topic questions, and helps users
understand the system's tennis-focused capabilities. It guides users toward effective
tennis-related queries and asks clarifying questions when needed.
"""

import time
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from src.config.settings import TennisConfig
    from src.utils.memory_manager import MemoryManager
except ImportError:
    from config.settings import TennisConfig
    from utils.memory_manager import MemoryManager


class ClarificationAgent:
    """
    Clarification agent that handles unclear queries and guides users toward tennis topics.
    
    Responsibilities:
    - Detect unclear or off-topic queries
    - Explain system capabilities and tennis focus
    - Ask clarifying questions to understand user intent
    - Suggest tennis-related alternatives when appropriate
    - Handle follow-up questions using conversation memory
    """
    
    def __init__(self, config: TennisConfig, memory_manager: MemoryManager):
        """
        Initialize the Clarification Agent.
        
        Args:
            config: Tennis system configuration
            memory_manager: Memory management instance for conversation context
        """
        self.config = config
        self.memory_manager = memory_manager
        self.llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.3,  # Slightly higher for more conversational responses
            api_key=config.openai_api_key
        )
        self.system_prompt = self._get_system_prompt()
    
    def handle_unclear_query(
        self, 
        user_query: str, 
        session_id: str,
        query_analysis: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle unclear, off-topic, or ambiguous user queries.
        
        Args:
            user_query: The user's question or request
            session_id: Session identifier for memory management
            query_analysis: Analysis from orchestrator about why this is unclear
            
        Returns:
            Dictionary containing clarification response and guidance
        """
        start_time = time.time()
        
        try:
            # Get conversation context for follow-up handling
            relevant_context = self.memory_manager.get_relevant_context(
                session_id, user_query, max_entries=5
            )
            
            # Check if this is a follow-up clarification that makes the intent clear
            reconstructed_query = self._try_reconstruct_intent(user_query, relevant_context)
            if reconstructed_query:
                # The intent is now clear - provide helpful guidance to ask the reconstructed query
                return {
                    "success": True,
                    "response": f"I understand now! It sounds like you're asking: '{reconstructed_query}'. Let me help you with that tennis question. Could you ask me that specific question?",
                    "response_type": "clarification_with_reconstruction",
                    "requires_follow_up": True,
                    "reconstructed_query": reconstructed_query,
                    "context_used": len(relevant_context),
                    "execution_time": time.time() - start_time,
                    "guidance_provided": True
                }
            
            # Create clarification prompt
            clarification_prompt = self._create_clarification_prompt(
                user_query, relevant_context, query_analysis
            )
            
            # Get LLM response
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=clarification_prompt)
            ]
            
            response = self.llm.invoke(messages)
            clarification_response = response.content
            
            # Store the clarification in memory
            self.memory_manager.add_interaction(
                session_id=session_id,
                user_query=user_query,
                response=clarification_response,
                response_type="clarification",
                metadata={
                    "requires_clarification": True,
                    "context_used": len(relevant_context),
                    "execution_time": time.time() - start_time
                }
            )
            
            return {
                "success": True,
                "response": clarification_response,
                "response_type": "clarification",
                "requires_follow_up": True,
                "context_used": len(relevant_context),
                "execution_time": time.time() - start_time,
                "guidance_provided": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": self._get_fallback_clarification(),
                "response_type": "clarification",
                "requires_follow_up": True,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "guidance_provided": True
            }
    
    def _create_clarification_prompt(
        self,
        user_query: str,
        relevant_context: List[Dict[str, Any]],
        query_analysis: Dict[str, Any] = None
    ) -> str:
        """Create a prompt for clarifying unclear queries."""
        
        # Build context from conversation history
        context_info = ""
        if relevant_context:
            context_info = "\n\nCONVERSATION CONTEXT:\n"
            for ctx in relevant_context[-3:]:  # Last 3 interactions
                context_info += f"Previous Query: {ctx.get('user_query', 'N/A')}\n"
                context_info += f"Response Type: {ctx.get('response_type', 'N/A')}\n"
                # Add key details from previous responses to help with follow-ups
                if ctx.get('response_type') == 'clarification':
                    context_info += f"Previous Response: {ctx.get('system_response', '')[:150]}...\n"
                context_info += "\n"
        
        # Check if this might be a follow-up clarification (like "I meant tennis player")
        follow_up_indicators = ["i meant", "i mean", "i was asking about", "no, i meant", "actually", "i'm asking about"]
        is_follow_up_clarification = any(indicator in user_query.lower() for indicator in follow_up_indicators)
        
        if is_follow_up_clarification and relevant_context:
            # This is likely a clarification of a previous unclear query
            context_info += "\nNOTE: This appears to be a clarification of a previous query. Try to combine the previous query with this clarification to understand the full intent.\n"
        
        # Add analysis information if available
        analysis_info = ""
        if query_analysis:
            tennis_entities = query_analysis.get("tennis_entities", [])
            if tennis_entities:
                analysis_info = f"\nDetected tennis entities: {', '.join(tennis_entities)}"
        
        prompt = f"""USER QUERY: "{user_query}"{context_info}{analysis_info}

ANALYSIS: This query needs clarification because it's unclear, off-topic, or requires more context.

YOUR TASK: Provide a helpful, contextual clarification response that:

1. **If this is a follow-up question** (like "when did this happen?" referring to previous tennis discussion):
   - Use the conversation context to understand what "this" refers to
   - Provide the specific information they're asking about
   - Be direct and helpful

1.5. **If this is a clarification of a previous unclear query** (like "I meant tennis player" after asking "who's the best"):
   - Combine the previous query with this clarification to understand the full intent
   - If the combined intent is clear (e.g., "who's the best" + "tennis player" = "who's the best tennis player"), provide a direct answer or redirect to the appropriate information
   - Don't ask for more clarification if the intent is now clear

2. **If this is incomplete but clearly heading toward tennis** (like "who's the best", "compare them", "what happened"):
   - Acknowledge what they're asking and intelligently guess the tennis context
   - Ask a specific clarifying question to complete their thought
   - Example: "Who's the best" → "Best at what? Are you asking about the current #1 tennis player, or perhaps comparing career achievements?"

3. **If this is vague but potentially tennis-related**:
   - Don't give generic instructions
   - Ask specific clarifying questions based on their actual words
   - Example: "tell me about tennis" → "What aspect of tennis interests you? Current players and rankings, historical matches, or maybe technique and rules?"

4. **If this is completely off-topic** (non-tennis):
   - Politely acknowledge their question but explain I'm tennis-focused
   - Ask if they have any tennis questions instead
   - Be brief and redirect to tennis

IMPORTANT: Always respond to their ACTUAL query with specific clarification, not generic help instructions. Be conversational and intelligent about what they might be asking."""

        return prompt
    
    def _try_reconstruct_intent(self, user_query: str, relevant_context: List[Dict[str, Any]]) -> Optional[str]:
        """
        Try to reconstruct the user's intent by combining current query with previous context.
        
        Args:
            user_query: Current user query (likely a clarification)
            relevant_context: Previous conversation context
            
        Returns:
            Reconstructed query if intent is clear, None otherwise
        """
        if not relevant_context:
            return None
        
        # Check if this looks like a clarification response
        clarification_indicators = ["i meant", "i mean", "i was asking about", "no, i meant", "actually", "i'm asking about"]
        is_clarification = any(indicator in user_query.lower() for indicator in clarification_indicators)
        
        if not is_clarification:
            return None
        
        # Get the most recent unclear query that needed clarification
        previous_unclear_query = None
        for ctx in relevant_context:
            if ctx.get('response_type') == 'clarification' or ctx.get('query_intent') == 'clarification_needed':
                previous_unclear_query = ctx.get('user_query')
                break
        
        if not previous_unclear_query:
            return None
        
        try:
            # Use LLM to reconstruct the intent
            reconstruction_prompt = f"""CONTEXT: A user previously asked: "{previous_unclear_query}"
I asked for clarification, and they responded: "{user_query}"

Try to reconstruct what the user originally wanted to ask. If the combination makes a clear tennis question, provide the reconstructed query. If not, return "UNCLEAR".

Examples:
- Previous: "who's the best" + Clarification: "I meant tennis player" → "who's the best tennis player"
- Previous: "compare them" + Clarification: "Federer and Nadal" → "compare Federer and Nadal"
- Previous: "when did it happen" + Clarification: "the Wimbledon final" → "when did the Wimbledon final happen"

Reconstructed query (or "UNCLEAR" if not clear):"""

            response = self.llm.invoke([
                SystemMessage(content="You reconstruct user intent from incomplete queries and clarifications. Be concise."),
                HumanMessage(content=reconstruction_prompt)
            ])
            
            reconstructed = response.content.strip()
            
            # Check if reconstruction was successful and looks like a tennis query
            if reconstructed.lower() != "unclear" and len(reconstructed) > 10:
                # Basic validation that it's a reasonable tennis query
                tennis_terms = ["tennis", "player", "match", "tournament", "ranking", "winner", "champion", "court", "serve"]
                if any(term in reconstructed.lower() for term in tennis_terms) or "who" in reconstructed.lower() or "what" in reconstructed.lower():
                    return reconstructed
            
            return None
            
        except Exception as e:
            # If LLM reconstruction fails, try simple text combination
            simple_reconstruction = f"{previous_unclear_query} {user_query}"
            if len(simple_reconstruction) < 100:  # Reasonable length
                return simple_reconstruction
            return None
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the clarification agent."""
        return """You are a Clarification Agent for a Tennis Intelligence System.

Your expertise is in:
- Understanding unclear user queries and asking the right follow-up questions
- Using conversation memory to resolve ambiguous references (like "when did this happen?")
- Explaining the tennis system's capabilities in a helpful way
- Guiding users toward productive tennis conversations

You have access to:
- Comprehensive tennis match database (2023-2025)
- Real-time tennis news and rankings
- Player statistics and career records
- Tournament results and schedules

Your communication style should be:
- Friendly and conversational
- Specific and actionable
- Tennis-focused but welcoming
- Patient with users learning the system

When handling follow-up questions, always check the conversation context first to understand what they're referring to."""
    
    def _get_fallback_clarification(self) -> str:
        """Get a fallback clarification message if the LLM fails."""
        return """I'm experiencing some technical difficulties, but I'm a tennis intelligence system ready to help! 

Could you try rephrasing your tennis question? I can help with player stats, match results, rankings, and tournament information.

What would you like to know about tennis?""" 