"""
Tennis Intelligence System - Main Entry Point
=============================================

Main CLI interface and demonstration of the tennis intelligence system.
This provides a working example of how the orchestrator and tools work together.
"""

import os
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


def create_session_id() -> str:
    """Create a unique session ID."""
    return f"session_{uuid.uuid4().hex[:8]}"

# Add src directory to Python path for proper imports
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Change directory to project root for relative file paths
os.chdir(current_dir)

try:
    from config.settings import TennisConfig, validate_config
    from agents.langgraph_orchestrator import LangGraphTennisOrchestrator  # LangGraph orchestrator
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure you're running from the eval_agent directory:")
    print("  cd eval_agent")
    print("  python tennis_system.py")
    sys.exit(1)


class TennisIntelligenceSystem:
    """
    Tennis Intelligence System using LangGraph orchestrator with official tool calling.
    
    This system uses LangGraph workflows for proper tool execution and tracking,
    enabling comprehensive evaluation with judgeval.
    """
    
    def __init__(self):
        """Initialize the tennis intelligence system with LangGraph orchestrator."""
        # Validate configuration
        config_issues = validate_config()
        if config_issues:
            print("❌ Configuration Issues:")
            for issue in config_issues:
                print(f"  • {issue}")
            sys.exit(1)
        
        # Initialize components
        self.config = TennisConfig()
        
        # Always use LangGraph orchestrator with built-in memory
        print("🚀 Using LangGraph orchestrator with built-in memory and official tool calling")
        self.orchestrator = LangGraphTennisOrchestrator(self.config)
        
        print("✅ Tennis Intelligence System initialized successfully")
        print(f"📊 Database: {self.config.database_path}")
        print(f"🤖 Model: {self.config.default_model}")
        print(f"🧠 LangGraph Orchestrator with Tool Calling")
    
    def process_query(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """
        Process a user query through the LangGraph tennis intelligence system.
        
        Args:
            user_query: The user's question or request
            session_id: Session identifier for memory management
            
        Returns:
            Dict containing the response and metadata with tool calling information
        """
        try:
            start_time = time.time()
            print(f"\n🚀 LangGraph Processing: '{user_query}'")
            
            # Process through LangGraph workflow
            result = self.orchestrator.process_query(user_query, session_id)
            
            # Add processing time
            result['processing_time'] = time.time() - start_time
            
            # Extract tool calling information (LangGraph tracks this automatically)
            tools_called = []
            if result.get('sql_data_used', False):
                tools_called.append('query_sql_database')
            if result.get('search_data_used', False):
                tools_called.append('online_search')
            
            result['tools_called'] = tools_called
            result['langgraph_used'] = True
            
            return result
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'response': error_msg,
                'confidence': 0.0,
                'sources': [],
                'error': True,
                'processing_time': time.time() - start_time,
                'tools_called': [],
                'langgraph_used': True
            }


def main():
    """Main function to run the tennis intelligence system demo."""
    print("🎾 Tennis Intelligence System")
    print("=" * 50)
    
    try:
        # Initialize system
        system = TennisIntelligenceSystem()
        session_id = create_session_id()
        
        print(f"\n🎾 Welcome to the Tennis Intelligence System!")
        print("Ask me anything about tennis - players, matches, rankings, etc.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("🎾 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Process the query
                start_time = time.time()
                result = system.process_query(user_input, session_id)
                end_time = time.time()
                
                # Display results
                print(f"\n🤖 Response:")
                print(f"   {result['response']}")
                print(f"\n📊 Metadata:")
                print(f"   • Confidence: {result['confidence']:.2f}")
                print(f"   • Sources: {', '.join(result['sources'])}")
                print(f"   • Tools Called: {', '.join(result.get('tools_called', []))}")
                print(f"   • Processing time: {end_time - start_time:.2f}s")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
                
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()