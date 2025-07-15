"""
Tennis Intelligence System - Classifier Models
==============================================

Pydantic models for tennis classification and query refinement with context awareness.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ConversationPair(BaseModel):
    """A single conversation pair (user query + system response)."""
    user_query: str = Field(..., description="User's question")
    system_response: str = Field(..., description="System's response")


class TennisClassificationResult(BaseModel):
    """Result of tennis-related classification."""
    is_tennis_related: bool = Field(..., description="Whether the query is tennis-related")
    reasoning: str = Field(..., description="Brief explanation of the classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in classification")


class QueryRefinementResult(BaseModel):
    """Result of query refinement with context."""
    is_tennis_related: bool = Field(..., description="Whether the refined query is tennis-related")
    refined_query: str = Field(..., description="Refined query incorporating context")
    reasoning: str = Field(..., description="Brief explanation of the refinement")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in refinement")
    context_used: bool = Field(..., description="Whether previous context was used for refinement")


class ContextAwareClassifier(BaseModel):
    """Context-aware tennis classifier with conversation history."""
    
    @staticmethod
    def classify_tennis_query(
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
        # This will be called by the actual classifier function
        pass
    
    @staticmethod
    def refine_query_with_context(
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
        # This will be called by the actual refiner function
        pass 