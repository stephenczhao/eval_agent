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
from typing import Dict, Any, List
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
    
    Database Constraints:
    - SQL database contains tennis data from 2023-01-01 to 2025-06-28 only
    - Queries for data after June 2025 and before 2023 should use online search
    - "Latest", "current", "right now" queries should prefer online search
    """
    
    # Database temporal constraints - precise date range
    DATABASE_START_DATE = "2023-01-01"
    DATABASE_END_DATE = "2025-06-28"
    
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
        
        # Add temporal constraints to config for orchestrator
        self.config.database_temporal_range = {
            'start_date': self.DATABASE_START_DATE,
            'end_date': self.DATABASE_END_DATE,
            'description': f"SQL database contains tennis data from {self.DATABASE_START_DATE} to {self.DATABASE_END_DATE} only. For data after June 2025 or 'latest/current' queries, prefer online search."
        }
        
        # Always use LangGraph orchestrator with built-in memory
        if self.debug:
            print("🚀 Using LangGraph orchestrator with built-in memory and official tool calling")
        self.orchestrator = LangGraphTennisOrchestrator(self.config, debug=debug)
        
        if self.debug:
            print("✅ Tennis Intelligence System initialized successfully")
            print(f"📊 Database: {self.config.database_path}")
            print(f"📅 Database Range: {self.DATABASE_START_DATE} to {self.DATABASE_END_DATE}")
            print(f"🤖 Model: {self.config.default_model}")
            print(f"🧠 LangGraph Orchestrator with Tool Calling")
    
    def _extract_query_years(self, query: str) -> List[int]:
        """
        Extract years mentioned in the query to help with temporal routing.
        
        Args:
            query: User query string
            
        Returns:
            List of years found in the query
        """
        import re
        # Find 4-digit years in the query (1900-2099)
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        return [int(year) for year in years]
    
    def _is_query_outside_database_range(self, query: str) -> bool:
        """
        Check if query asks for data outside the database temporal range.
        
        Args:
            query: User query string
            
        Returns:
            True if query mentions years outside our database range
        """
        years = self._extract_query_years(query)
        if not years:
            return False
        
        # Check if any mentioned year is outside our database range (2023-2025)
        for year in years:
            if year < 2023 or year > 2025:
                return True
        return False

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
            
            # Don't add date prefixes to avoid search issues with future dates
            enhanced_query = user_query
            
            # Check for temporal constraints using original query
            outside_db_range = self._is_query_outside_database_range(user_query)
            query_years = self._extract_query_years(user_query)
            
            if self.debug and outside_db_range:
                print(f"⚠️  Query mentions years outside database range ({self.DATABASE_START_DATE} to {self.DATABASE_END_DATE})")
                if query_years:
                    print(f"📅 Query years: {query_years}")
                print(f"🌐 Note: LLM router will determine best data source")
            
            if self.debug:
                print(f"\n🚀 LangGraph Processing: '{user_query}'")
                # print(f"🕒 Date prefix: [{datetime_str}]")
            # else:
            #     print(f"\n🚀 LangGraph Processing: '{user_query}'")
            
            # Process through LangGraph workflow with enhanced query
            result = self.orchestrator.process_query(enhanced_query, session_id)
            
            # Add processing time and temporal analysis
            result['processing_time'] = time.time() - start_time
            result['query_years'] = query_years
            result['outside_database_range'] = outside_db_range
            result['database_temporal_range'] = f"{self.DATABASE_START_DATE} to {self.DATABASE_END_DATE}"
            
            # Extract tool calling information (LangGraph tracks this automatically)
            tools_called = []
            if result.get('sql_data_used', False):
                tools_called.append('query_sql_database')
            if result.get('search_data_used', False):
                tools_called.append('online_search')
            
            result['tools_called'] = tools_called
            result['langgraph_used'] = True
            
            # Add routing recommendation based on temporal analysis
            if outside_db_range and result.get('sql_data_used', False):
                result['routing_note'] = f"Query asked for data from {query_years} but used SQL database (2023-2025 only)"
            elif outside_db_range and result.get('search_data_used', False):
                result['routing_note'] = f"Correctly used online search for data from {query_years} (outside database range)"
            
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
    debug_mode = os.environ.get('TENNIS_DEBUG', 'False').lower() == 'true'
    
    try:
        # Initialize system
        system = TennisIntelligenceSystem(debug=debug_mode)
        session_id = create_session_id()
        
        print(f"\n🎾 Welcome to the Tennis Intelligence System!")
        if debug_mode:
            print("🐛 Debug mode enabled - showing detailed processing steps")
        print("Ask me anything about tennis - players, matches, rankings, etc.")
        print(f"📅 Database covers: {TennisIntelligenceSystem.DATABASE_START_DATE} to {TennisIntelligenceSystem.DATABASE_END_DATE}")
        print("🌐 For latest/current rankings or post-June 2025 data, I'll search online")
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
                    print(f"   • Database Range: {result.get('database_temporal_range', 'Unknown')}")
                    
                    # Show temporal analysis if years were detected
                    if result.get('query_years'):
                        print(f"   • Query Years: {result['query_years']}")
                        if result.get('outside_database_range'):
                            print(f"   • ⚠️  Outside Database Range: Yes")
                        else:
                            print(f"   • ✅ Within Database Range: Yes")
                    
                    # Show routing notes if any
                    if result.get('routing_note'):
                        print(f"   • Routing Note: {result['routing_note']}")
                
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