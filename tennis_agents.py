"""
Tennis Agent System - Main Entry Point
=============================================

Main CLI interface and demonstration of the tennis intelligence system.
This provides a working example of how the orchestrator and tools work together.
"""

import os
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from src.config.settings import TennisConfig, validate_config
from src.agents.langgraph_orchestrator import LangGraphTennisOrchestrator  # LangGraph orchestrator


def create_session_id() -> str:
    """Create a unique session ID."""
    return f"session_{uuid.uuid4().hex[:8]}"



class TennisAgentSystem:
    """
    Tennis Intelligence System using LangGraph orchestrator with official tool calling.
    
    This system uses LangGraph workflows for proper tool execution and tracking,
    enabling comprehensive evaluation with judgeval.
    
    The system provides context about data sources and temporal ranges, 
    but lets the LLM make all routing and decision choices.
    """
    
    def __init__(self, debug: bool = False):
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
        self.debug = debug
        
        # Add database context information for LLM reasoning (not rules)
        self.database_context = {
            'temporal_coverage': '2023-01-01 to 2025-06-28',
            'description': 'SQL database contains comprehensive tennis match data including player stats, tournament results, and rankings for the specified date range.',
            'current_date': datetime.now().strftime('%Y-%m-%d'),
            'data_sources': {
                'sql_database': 'Historical match data, player statistics, head-to-head records, tournament results',
                'web_search': 'Current rankings, recent news, live updates, breaking tennis news'
            }
        }
        
        # Always use LangGraph orchestrator with built-in memory
        if self.debug:
            print("🚀 Using LangGraph orchestrator with built-in memory and official tool calling")
        self.orchestrator = LangGraphTennisOrchestrator(self.config, debug=debug)
        
        if self.debug:
            print("✅ Tennis Intelligence System initialized successfully")
            print(f"📊 Database: {self.config.database_path}")
            print(f"📅 Database Coverage: 2023-01-01 to 2025-06-28")
            print(f"🤖 Model: {self.config.default_model}")
            print(f"🧠 LangGraph Orchestrator with Tool Calling")

    def process_query(self, user_query: str, session_id: str, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """
        Process a user query through the LangGraph tennis intelligence system.
        
        Args:
            user_query: The user's question or request
            session_id: Session identifier for memory management
            callbacks: Optional list of LangChain callbacks for trace capture
            
        Returns:
            Dict containing the response and metadata with tool calling information
        """
        try:
            start_time = time.time()
        
            
            if self.debug:
                if callbacks:
                    print(f"📊 Trace capture enabled with {len(callbacks)} callback(s)")
            
            # Process through LangGraph workflow with callbacks
            result = self.orchestrator.process_query(user_query, session_id, callbacks=callbacks)
            
            # Add processing time and context metadata
            result['processing_time'] = time.time() - start_time
            result['database_context'] = self.database_context
            
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
    
    # Add debug mode support - can be set here or via environment variable
    import os
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    try:
        # Initialize system
        system = TennisAgentSystem(debug=debug_mode)
        session_id = create_session_id()
        
        print(f"\n🎾 Welcome to the Tennis Intelligence System!")
        if debug_mode:
            print("🐛 Debug mode enabled - showing detailed processing steps")
        print("Ask me anything about tennis - players, matches, rankings, etc.")
        print(f"📅 Database covers: 2023-01-01 to 2025-06-28")
        print("🌐 I have access to both historical data and current web information")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("🎾 Ask me something about tennis: ").strip()
                
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
                print(f"\n🤖 Answer:")
                print(f"   {result['response']}")
                
                # Only show metadata in debug mode
                if debug_mode:
                    print(f"\n📊 Metadata:")
                    print(f"   • Confidence: {result['confidence']:.2f}")
                    print(f"   • Sources: {', '.join(result['sources'])}")
                    print(f"   • Tools Called: {', '.join(result.get('tools_called', []))}")
                    print(f"   • Processing time: {end_time - start_time:.2f}s")
                    
                    if result.get('database_context'):
                        db_context = result['database_context']
                        print(f"   • Database Coverage: {db_context['temporal_coverage']}")
                        print(f"   • Current Date: {db_context['current_date']}")
                
                print()  # Add spacing
                
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