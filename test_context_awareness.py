#!/usr/bin/env python3
"""
Test script for context awareness improvements.
"""

import sys
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

from utils.simple_memory_manager import SimpleMemoryManager, create_session_id
from utils.context_aware_classifier import classify_tennis_query, refine_query_with_context
from models.classifier_models import ConversationPair


def test_context_awareness():
    """Test the context awareness improvements."""
    print("🧪 Testing Context Awareness Improvements")
    print("=" * 50)
    
    # Initialize memory manager
    memory_manager = SimpleMemoryManager()
    session_id = create_session_id()
    
    # Test 1: Initial query
    print("\n📋 Test 1: Initial Query")
    query1 = "who's the best female player?"
    result1 = refine_query_with_context(query1, [])
    print(f"Query: '{query1}'")
    print(f"Tennis-related: {result1.is_tennis_related}")
    print(f"Refined: '{result1.refined_query}'")
    print(f"Reasoning: {result1.reasoning}")
    
    # Simulate storing conversation
    memory_manager.store_conversation(
        session_id=session_id,
        user_query=query1,
        system_response="Aryna Sabalenka is currently the best female tennis player, ranked #1."
    )
    
    # Test 2: Follow-up query with context
    print("\n📋 Test 2: Follow-up Query with Context")
    query2 = "what about men?"
    conversation_history = memory_manager.get_conversation_history(session_id)
    result2 = refine_query_with_context(query2, conversation_history)
    print(f"Query: '{query2}'")
    print(f"Context: {len(conversation_history)} conversation pairs")
    print(f"Tennis-related: {result2.is_tennis_related}")
    print(f"Refined: '{result2.refined_query}'")
    print(f"Reasoning: {result2.reasoning}")
    print(f"Context used: {result2.context_used}")
    
    # Test 3: Another follow-up
    memory_manager.store_conversation(
        session_id=session_id,
        user_query=query2,
        system_response="Jannik Sinner is currently the best male tennis player, ranked #1."
    )
    
    print("\n📋 Test 3: Another Follow-up Query")
    query3 = "who's second?"
    conversation_history = memory_manager.get_conversation_history(session_id)
    result3 = refine_query_with_context(query3, conversation_history)
    print(f"Query: '{query3}'")
    print(f"Context: {len(conversation_history)} conversation pairs")
    print(f"Tennis-related: {result3.is_tennis_related}")
    print(f"Refined: '{result3.refined_query}'")
    print(f"Reasoning: {result3.reasoning}")
    print(f"Context used: {result3.context_used}")
    
    # Test 4: Non-tennis query with tennis context
    print("\n📋 Test 4: Non-tennis Query with Tennis Context")
    query4 = "what's the weather like?"
    result4 = refine_query_with_context(query4, conversation_history)
    print(f"Query: '{query4}'")
    print(f"Tennis-related: {result4.is_tennis_related}")
    print(f"Refined: '{result4.refined_query}'")
    print(f"Reasoning: {result4.reasoning}")
    
    # Test 5: Clear follow-up
    print("\n📋 Test 5: Clear Follow-up")
    query5 = "and clay courts?"
    result5 = refine_query_with_context(query5, conversation_history)
    print(f"Query: '{query5}'")
    print(f"Tennis-related: {result5.is_tennis_related}")
    print(f"Refined: '{result5.refined_query}'")
    print(f"Reasoning: {result5.reasoning}")
    print(f"Context used: {result5.context_used}")
    
    print("\n✅ Context awareness tests completed!")


if __name__ == "__main__":
    test_context_awareness() 