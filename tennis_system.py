"""
Tennis Intelligence System - Main Entry Point
=============================================

Main CLI interface and demonstration of the tennis intelligence system.
This provides a working example of how the orchestrator and tools work together.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Add src directory to Python path for proper imports
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Change directory to project root for relative file paths
os.chdir(current_dir)

try:
    from config.settings import TennisConfig, validate_config
    from utils.memory_manager import MemoryManager, create_session_id
    from agents.orchestrator import OrchestratorAgent
    from tools.sql_tools import execute_sql_query, generate_sql_query
    from tools.search_tools import tavily_search_tool, summarize_search_results, optimize_search_query
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
    Main system that demonstrates the orchestrator working with specialized tools.
    
    This is a simplified version showing the core functionality before 
    implementing the full LangGraph workflow.
    """
    
    def __init__(self):
        """Initialize the tennis intelligence system."""
        # Validate configuration
        config_issues = validate_config()
        if config_issues:
            print("❌ Configuration Issues:")
            for issue in config_issues:
                print(f"  • {issue}")
            sys.exit(1)
        
        # Initialize components
        self.config = TennisConfig()
        self.memory_manager = MemoryManager()
        self.orchestrator = OrchestratorAgent(self.config, self.memory_manager)
        
        print("✅ Tennis Intelligence System initialized successfully")
        print(f"📊 Database: {self.config.database_path}")
        print(f"🤖 Model: {self.config.default_model}")
        print(f"🧠 Agents: Orchestrator, SQL, Search")
    
    def _is_tennis_related(self, user_query: str) -> bool:
        """
        Simple LLM classifier to determine if the query is tennis-related.
        
        Args:
            user_query: The user's question
            
        Returns:
            True if tennis-related, False otherwise
        """
        try:
            llm = ChatOpenAI(model=self.config.default_model, temperature=0.1)
            
            classifier_prompt = f"""Is this query related to tennis? Answer with only YES or NO.

Query: "{user_query}"

CONTEXT: This is a tennis intelligence system that answers questions about tennis players, matches, tournaments, rankings, and related topics.

Consider tennis-related if it mentions or asks about:
- Tennis players (by name or in general: "who's the best player", "top player", "player rankings")
- Tennis tournaments ("latest tournament", "who won", "tournament results", "Wimbledon", "US Open", etc.)
- Tennis matches ("match results", "who played", "match statistics")
- Tennis rankings ("best player", "top ranked", "world ranking", "#1 player")
- Tennis techniques, rules, equipment, coaching
- Tennis statistics, records, results, scores
- Tennis news, schedules, injuries
- General sports questions in a tennis context ("who's the best", "who won", "latest results" in a tennis system)

IMPORTANT: Be generous with tennis interpretation. If someone asks "who's the best player?" in a tennis system, that's clearly asking about tennis players. Similarly, "who won the latest tournament?" is asking about tennis tournaments.

Examples:
- "who's the best player?" → YES (asking about tennis players)
- "who won the latest tournament?" → YES (asking about tennis tournament)
- "what's Federer's record?" → YES (tennis player statistics)
- "who's ranked #1?" → YES (tennis rankings)
- "latest match results" → YES (tennis matches)
- "what's the weather like?" → NO (not tennis-related)
- "help me with math" → NO (not tennis-related)

Answer YES for tennis queries, NO for clearly non-tennis topics.

Response (YES or NO only):"""

            response = llm.invoke([
                SystemMessage(content="You are a tennis topic classifier. Be generous with tennis context. Respond with only YES or NO."),
                HumanMessage(content=classifier_prompt)
            ])
            
            return response.content.strip().upper() == "YES"
            
        except Exception as e:
            print(f"⚠️ Tennis classifier error: {e}")
            # Default to allowing the query if classifier fails
            return True
    
    def process_query(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """
        Process a user query through the tennis intelligence system.
        
        Args:
            user_query: The user's question or request
            session_id: Session identifier for memory management
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            start_time = time.time()
            
            # Step 1: Check if query is tennis-related
            print(f"\n🎾 Checking if query is tennis-related: '{user_query}'")
            if not self._is_tennis_related(user_query):
                non_tennis_response = "I'm designed specifically to answer tennis-related questions. Please ask me about tennis players, matches, tournaments, rankings, or tennis techniques."
                
                execution_time = time.time() - start_time
                print(f"❌ Non-tennis query detected")
                
                return {
                    'response': non_tennis_response,
                    'confidence': 1.0,
                    'sources': ['Tennis Classifier'],
                    'action': 'non_tennis_rejection',
                    'processing_time': execution_time
                }
            
            print(f"✅ Tennis-related query confirmed")
            
            # Step 2: Orchestrator analyzes query and determines routing
            print(f"\n🧠 Orchestrator analyzing: '{user_query}'")
            routing_result = self.orchestrator.analyze_and_route(
                user_query, 
                session_id
            )
            
            # Extract routing decisions
            routing_decision = routing_result.get('routing_decision', {})
            sql_needed = routing_decision.get('sql_needed', False)
            search_needed = routing_decision.get('search_needed', False)
            
            # Determine action for display
            if sql_needed and search_needed:
                action = "sql_and_search"
            elif sql_needed:
                action = "sql_only"
            elif search_needed:
                action = "search_only"
            else:
                # Default to search if nothing is determined
                action = "search_only"
                search_needed = True
            
            print(f"🎯 Routing decision: {action}")
            if routing_decision.get('reasoning'):
                print(f"💭 Reasoning: {routing_decision['reasoning']}")
            
            # Step 3: Execute appropriate agents based on routing
            sql_results = None
            search_results = None
            
            if sql_needed:
                sql_results = self._execute_sql_agent(user_query, routing_result)
                
            if search_needed:
                search_results = self._execute_search_agent(user_query, routing_result)
            
            # Step 4: Synthesize final response
            response = self._synthesize_response(
                user_query, sql_results, search_results, routing_result
            )
            
            # Store complete conversation in memory
            execution_time = time.time() - start_time
            
            # Debug: Show what we're storing in memory
            entities_to_store = routing_result.get('extracted_entities', [])
            intent_to_store = routing_result.get('query_analysis', {}).get('intent', 'unknown')
            
            print(f"🧠 Storing conversation in memory:")
            print(f"   • Session: {session_id}")
            print(f"   • Entities: {entities_to_store}")
            print(f"   • Intent: {intent_to_store}")
            print(f"   • Confidence: {response.get('confidence', 0.0):.2f}")
            
            self.memory_manager.store_conversation(
                session_id=session_id,
                user_query=user_query,
                system_response=response['response'],
                sources_used=response.get('sources', []),
                confidence_score=response.get('confidence', 0.0),
                execution_time=execution_time,
                tennis_entities=entities_to_store,
                query_intent=intent_to_store
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'response': error_msg,
                'confidence': 0.0,
                'sources': [],
                'error': True
            }
    
    def _execute_sql_agent(self, user_query: str, routing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL database queries based on the user query."""
        print("🗄️  Executing SQL Agent...")
        
        try:
            # Generate SQL query using LLM text-to-SQL
            print("🧠 Generating SQL query from natural language...")
            sql_generation = generate_sql_query(user_query)
            
            if not sql_generation.get('success', False):
                print(f"❌ SQL generation failed: {sql_generation.get('error', 'Unknown error')}")
                return {
                    'query': None,
                    'results': [],
                    'row_count': 0,
                    'success': False,
                    'error': f"SQL generation failed: {sql_generation.get('error', 'Unknown error')}"
                }
            
            sql_query = sql_generation.get('sql_query')
            print(f"✨ Generated SQL: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}")
            
            # Execute the generated query
            results = execute_sql_query.invoke({"query": sql_query})
            
            print(f"📊 SQL query executed: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}")
            
            # Handle the results based on the execute_sql_query response format
            if results.get('success', False):
                row_count = results.get('row_count', 0)
                print(f"📈 Returned {row_count} rows")
                
                return {
                    'query': sql_query,
                    'results': results,
                    'row_count': row_count,
                    'success': True,
                    'generated_sql': True
                }
            else:
                print(f"❌ Query execution failed: {results.get('error', 'Unknown error')}")
                return {
                    'query': sql_query,
                    'results': results,
                    'row_count': 0,
                    'success': False,
                    'error': results.get('error', 'Query execution failed'),
                    'generated_sql': True
                }
            
        except Exception as e:
            print(f"❌ SQL Agent error: {e}")
            return {
                'query': sql_query if 'sql_query' in locals() else None,
                'results': [],
                'row_count': 0,
                'success': False,
                'error': str(e),
                'generated_sql': False
            }
    
    def _execute_search_agent(self, user_query: str, routing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search based on the user query."""
        print("🌐 Executing Search Agent...")
        
        try:
            # Extract search query from routing result or use original query
            routing_decision = routing_result.get('routing_decision', {})
            initial_search_query = routing_decision.get('search_query', user_query)
            
            # Step 1: Optimize the search query using the search query expert
            print("🧠 Optimizing search query...")
            optimization_result = optimize_search_query.invoke({
                "user_query": initial_search_query,
                "context": f"Tennis query routing: {routing_decision.get('reasoning', '')}"
            })
            
            # Use optimized query if successful, fallback to original if failed
            if optimization_result.get('success', False):
                optimized_query = optimization_result.get('optimized_query', initial_search_query)
                print(f"✨ Query optimized: '{initial_search_query}' → '{optimized_query}'")
            else:
                optimized_query = initial_search_query
                print(f"⚠️  Query optimization failed, using original: '{initial_search_query}'")
            
            # Step 2: Perform search with optimized query
            search_results = tavily_search_tool.invoke({"query": optimized_query})
            
            # Step 3: Summarize results
            summary = summarize_search_results.invoke({
                "search_results": search_results, 
                "focus_context": user_query
            })
            
            print(f"🔍 Final search query: {optimized_query}")
            # Extract results from the tool's structured output
            actual_results = search_results.get('raw_results', {}).get('results', [])
            print(f"📑 Found {len(actual_results)} results")
            
            return {
                'query': optimized_query,
                'original_query': initial_search_query,
                'optimization_result': optimization_result,
                'results': search_results,
                'summary': summary,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Search Agent error: {e}")
            return {
                'query': initial_search_query if 'initial_search_query' in locals() else None,
                'original_query': user_query,
                'optimization_result': {},
                'results': {},
                'summary': "",
                'success': False,
                'error': str(e)
            }
    
    def _synthesize_response(
        self,
        user_query: str,
        sql_results: Dict[str, Any],
        search_results: Dict[str, Any],
        routing_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize a final response from SQL and search results."""
        print("🧮 Synthesizing final response...")
        
        sources = []
        confidence = 0.8  # Base confidence
        final_response = ""
        
        # Print technical details for monitoring
        if sql_results and sql_results.get('success'):
            print(f"📊 SQL: Found {sql_results.get('row_count', 0)} database records")
            sources.append('Tennis Database')
            confidence += 0.1
        
        if search_results and search_results.get('success'):
            tavily_results = search_results.get('results', {})
            raw_results = tavily_results.get('raw_results', {})
            results_count = len(raw_results.get('results', []))
            if results_count > 0:
                # Show query optimization info
                if search_results.get('optimization_result', {}).get('optimization_applied'):
                    original_q = search_results.get('original_query', '')
                    optimized_q = search_results.get('query', '')
                    print(f"🌐 Search: Found {results_count} web results (query optimized)")
                    print(f"   Original: '{original_q}' → Optimized: '{optimized_q}'")
                else:
                    print(f"🌐 Search: Found {results_count} web results")
                sources.append('Web Search')
                confidence += 0.1
        
        # Use LLM to synthesize results - just pass everything to the LLM!
        
        llm = ChatOpenAI(model=self.config.default_model, temperature=0.3)
        
        # Get current datetime context
        current_datetime = datetime.now()
        current_date_str = current_datetime.strftime("%Y-%m-%d %H:%M")
        current_year = current_datetime.year
        
        # Prepare all available information for the LLM
        info_for_llm = f"User Question: {user_query}\n\n"
        
        # Add search results if available
        if search_results and search_results.get('success'):
            # The search agent returns results nested under 'results' key
            tavily_results = search_results.get('results', {})
            raw_results = tavily_results.get('raw_results', {})
            
            if raw_results and raw_results.get('results'):
                info_for_llm += "Web Search Results:\n"
                for i, result in enumerate(raw_results['results'][:5], 1):
                    info_for_llm += f"{i}. {result.get('title', 'No title')}\n"
                    info_for_llm += f"   {result.get('content', 'No content')}\n\n"
                
                # Debug: Print what we're sending to the LLM
                print(f"🔍 Debug: Sending {len(raw_results['results'])} search results to LLM")
                print(f"📝 Debug: First result title: {raw_results['results'][0].get('title', 'N/A')}")
            else:
                print("⚠️  Debug: No search results found in nested structure")
                print(f"🔍 Debug: search_results keys: {list(search_results.keys())}")
                print(f"🔍 Debug: tavily_results keys: {list(tavily_results.keys()) if tavily_results else 'None'}")
                print(f"🔍 Debug: raw_results keys: {list(raw_results.keys()) if raw_results else 'None'}")
        else:
            print("⚠️  Debug: No successful search_results provided to synthesis")
        
        # Add SQL results if available  
        if sql_results and sql_results.get('success') and sql_results.get('row_count', 0) > 0:
            info_for_llm += f"Database Results: Found {sql_results['row_count']} relevant records\n\n"
        
        # Enhanced synthesis prompt with temporal context
        synthesis_prompt = f"""{info_for_llm}

CURRENT DATE AND CONTEXT:
- Today's date: {current_date_str}
- Current year: {current_year}
- Available database: Tennis matches and player data from 2023-{current_year}
- You have access to comprehensive tennis data through {current_year}

IMPORTANT INSTRUCTIONS:
- You are answering in {current_year}, so questions about {current_year} events are NOT in the future
- The tennis database contains actual match data through {current_year}
- Do NOT refuse to answer questions about {current_year} events - this data is available
- Do NOT mention training cutoffs or refuse based on dates - you have current data
- If asking about {current_year} tennis matches/stats, use the database results provided
- Be confident in providing information about tennis events through {current_year}

Based on the above retrieved information, provide a helpful, conversational and effective response to the user's question: {user_query}

Focus on being factual and helpful rather than expressing uncertainty about temporal limitations."""
        
        try:
            messages = [
                SystemMessage(content=f"You are a helpful tennis expert assistant with access to current tennis data through {current_year}. Today is {current_date_str}. Answer questions confidently using the provided data."),
                HumanMessage(content=synthesis_prompt)
            ]
            response = llm.invoke(messages)
            final_response = response.content
            confidence = 0.9
        except Exception as e:
            print(f"⚠️  LLM synthesis failed: {e}")
            final_response = "I found tennis information but had trouble processing it. Could you try asking again?"
            confidence = 0.3
        
        return {
            'response': final_response,
            'confidence': min(confidence, 1.0),
            'sources': sources,
            'sql_data': sql_results,
            'search_data': search_results,
            'routing': routing_result
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