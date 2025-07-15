"""
Tennis Intelligence System - Context-Aware Classifier
====================================================

Context-aware tennis classifier and query refiner that uses conversation history
to properly classify and refine user queries.
"""

import json
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

try:
    from src.models.classifier_models import ConversationPair, TennisClassificationResult, QueryRefinementResult
    from src.config.settings import TennisConfig
except ImportError:
    from models.classifier_models import ConversationPair, TennisClassificationResult, QueryRefinementResult
    from config.settings import TennisConfig


class ContextAwareTennisClassifier:
    """
    Context-aware tennis classifier that uses conversation history to properly
    classify tennis-related queries and refine ambiguous queries.
    """
    
    def __init__(self, config: TennisConfig = None):
        """Initialize the classifier."""
        self.config = config or TennisConfig()
        self.llm = ChatOpenAI(
            model=self.config.default_model,
            temperature=0.1,
            max_tokens=200,  # Short responses for classification
            api_key=self.config.openai_api_key
        )
    
    def classify_tennis_query(
        self,
        user_query: str,
        conversation_history: List[ConversationPair] = None
    ) -> TennisClassificationResult:
        """
        Classify if a query is tennis-related using conversation context.
        
        Args:
            user_query: The user's current question
            conversation_history: Previous conversation pairs (up to 5)
            
        Returns:
            TennisClassificationResult with classification and reasoning
        """
        try:
            # Build context from conversation history
            context_str = self._build_context_string(conversation_history)
            
            # Create classification prompt
            prompt = f"""You are a tennis topic classifier. Determine if the user's query is tennis-related.

{context_str}

CURRENT QUERY: "{user_query}"

CLASSIFICATION RULES:
- Tennis-related: queries about tennis players, matches, tournaments, rankings, techniques, equipment, rules, coaching, statistics
- Context matters: if previous conversation was about tennis, follow-up questions like "what about men?", "and clay courts?", "who's second?" are tennis-related
- Be generous with context: if user is continuing a tennis conversation, assume tennis context

Return JSON:
{{
    "is_tennis_related": true/false,
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0
}}"""

            # Get classification
            response = self.llm.invoke([
                SystemMessage(content="You are a tennis topic classifier. Always return valid JSON."),
                HumanMessage(content=prompt)
            ])
            
            # Parse JSON response
            result_dict = self._parse_json_response(response.content)
            
            return TennisClassificationResult(
                is_tennis_related=result_dict.get("is_tennis_related", False),
                reasoning=result_dict.get("reasoning", "Classification failed"),
                confidence=result_dict.get("confidence", 0.5)
            )
            
        except Exception as e:
            print(f"⚠️  Tennis classification failed: {e}")
            return TennisClassificationResult(
                is_tennis_related=False,
                reasoning=f"Classification error: {str(e)}",
                confidence=0.0
            )
    
    def refine_query_with_context(
        self,
        user_query: str,
        conversation_history: List[ConversationPair] = None
    ) -> QueryRefinementResult:
        """
        Refine a user query using conversation context.
        
        Args:
            user_query: The user's current question
            conversation_history: Previous conversation pairs (up to 5)
            
        Returns:
            QueryRefinementResult with refined query and tennis classification
        """
        try:
            # Build context from conversation history
            context_str = self._build_context_string(conversation_history)
            
            # Create refinement prompt
            prompt = f"""You are a tennis query refiner. Refine the user's query using conversation context.

{context_str}

CURRENT QUERY: "{user_query}"

REFINEMENT RULES:
- If query is complete and clear, return it as-is
- If query is incomplete/ambiguous, use context to clarify
- Examples:
  * "what about men?" + tennis context → "who's the best men's tennis player?"
  * "and clay courts?" + player context → "what about [player]'s clay court performance?"
  * "who's second?" + ranking context → "who's the second-ranked tennis player?"
- Maintain tennis focus if context suggests tennis

Return JSON:
{{
    "is_tennis_related": true/false,
    "refined_query": "refined query string",
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0,
    "context_used": true/false
}}"""

            # Get refinement
            response = self.llm.invoke([
                SystemMessage(content="You are a tennis query refiner. Always return valid JSON."),
                HumanMessage(content=prompt)
            ])
            
            # Parse JSON response
            result_dict = self._parse_json_response(response.content)
            
            return QueryRefinementResult(
                is_tennis_related=result_dict.get("is_tennis_related", False),
                refined_query=result_dict.get("refined_query", user_query),
                reasoning=result_dict.get("reasoning", "Refinement failed"),
                confidence=result_dict.get("confidence", 0.5),
                context_used=result_dict.get("context_used", False)
            )
            
        except Exception as e:
            print(f"⚠️  Query refinement failed: {e}")
            return QueryRefinementResult(
                is_tennis_related=False,
                refined_query=user_query,
                reasoning=f"Refinement error: {str(e)}",
                confidence=0.0,
                context_used=False
            )
    
    def _build_context_string(self, conversation_history: List[ConversationPair]) -> str:
        """Build context string from conversation history."""
        if not conversation_history:
            return "CONTEXT: No previous conversation."
        
        context_lines = ["PREVIOUS CONVERSATION:"]
        for i, pair in enumerate(conversation_history[-5:], 1):  # Last 5 pairs
            # Truncate long responses for context
            response_snippet = pair.system_response[:100]
            if len(pair.system_response) > 100:
                response_snippet += "..."
            
            context_lines.append(f"{i}. User: {pair.user_query}")
            context_lines.append(f"   System: {response_snippet}")
        
        return "\n".join(context_lines)
    
    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON response from LLM."""
        try:
            # Try direct JSON parsing
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                try:
                    return json.loads(response_text[start_idx:end_idx])
                except json.JSONDecodeError:
                    pass
            
            # Fallback - return default structure
            return {
                "is_tennis_related": False,
                "reasoning": "Failed to parse LLM response",
                "confidence": 0.0,
                "refined_query": "",
                "context_used": False
            }


def classify_tennis_query(
    user_query: str,
    conversation_history: List[ConversationPair] = None,
    config: TennisConfig = None
) -> TennisClassificationResult:
    """
    Convenience function to classify tennis queries.
    
    Args:
        user_query: The user's current question
        conversation_history: Previous conversation pairs (up to 5)
        config: Tennis system configuration
        
    Returns:
        TennisClassificationResult with classification and reasoning
    """
    classifier = ContextAwareTennisClassifier(config)
    return classifier.classify_tennis_query(user_query, conversation_history)


def refine_query_with_context(
    user_query: str,
    conversation_history: List[ConversationPair] = None,
    config: TennisConfig = None
) -> QueryRefinementResult:
    """
    Convenience function to refine queries with context.
    
    Args:
        user_query: The user's current question
        conversation_history: Previous conversation pairs (up to 5)
        config: Tennis system configuration
        
    Returns:
        QueryRefinementResult with refined query and tennis classification
    """
    classifier = ContextAwareTennisClassifier(config)
    return classifier.refine_query_with_context(user_query, conversation_history) 