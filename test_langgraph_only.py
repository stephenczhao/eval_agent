#!/usr/bin/env python3
"""
Test script to verify LangGraph-only tennis intelligence system works correctly.
"""

import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

from tennis_agents import TennisIntelligenceSystem
from utils.simple_memory_manager import create_session_id


def test_tennis_system():
    """Test the LangGraph-only tennis intelligence system."""
    print("🧪 Testing LangGraph-only Tennis Intelligence System")
    print("=" * 60)
    
    try:
        # Initialize system
        print("1. Initializing system...")
        system = TennisIntelligenceSystem()
        session_id = create_session_id()
        print("   ✅ System initialized successfully")
        
        # Test query
        test_query = "Who won the most recent Wimbledon men's singles title?"
        print(f"\n2. Testing query: '{test_query}'")
        
        # Process query
        result = system.process_query(test_query, session_id)
        
        # Verify results
        print(f"\n3. Results:")
        print(f"   Response: {result.get('response', 'No response')}")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")
        print(f"   Sources: {result.get('sources', [])}")
        print(f"   Tools Called: {result.get('tools_called', [])}")
        print(f"   LangGraph Used: {result.get('langgraph_used', False)}")
        print(f"   Processing Time: {result.get('processing_time', 0):.2f}s")
        
        # Verify tool calling
        assert result.get('langgraph_used', False), "LangGraph should be used"
        assert isinstance(result.get('tools_called', []), list), "Tools called should be a list"
        
        print(f"\n✅ Test completed successfully!")
        print(f"   - LangGraph orchestrator: ✓")
        print(f"   - Tool calling tracking: ✓")
        print(f"   - Response generation: ✓")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_tennis_system()
    sys.exit(0 if success else 1) 