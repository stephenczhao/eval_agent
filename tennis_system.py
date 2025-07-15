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
    from utils.simple_memory_manager import SimpleMemoryManager, create_session_id
    from utils.context_aware_classifier import classify_tennis_query, refine_query_with_context
    from agents.orchestrator import OrchestratorAgent
    from tools.sql_tools import execute_sql_query, generate_sql_query, interpret_sql_results
    from tools.search_tools import tavily_search_tool, interpret_search_results, optimize_search_query
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
        self.memory_manager = SimpleMemoryManager()
        self.orchestrator = OrchestratorAgent(self.config, self.memory_manager)
        
        print("✅ Tennis Intelligence System initialized successfully")
        print(f"📊 Database: {self.config.database_path}")
        print(f"🤖 Model: {self.config.default_model}")
        print(f"🧠 Agents: Orchestrator, SQL, Search")
    
    def _classify_and_refine_query(self, user_query: str, session_id: str) -> tuple[bool, str, str]:
        """
        Context-aware tennis classifier and query refiner.
        
        Args:
            user_query: The user's question
            session_id: Session ID for conversation history
            
        Returns:
            Tuple of (is_tennis_related, refined_query, reasoning)
        """
        try:
            # Get conversation history
            conversation_history = self.memory_manager.get_conversation_history(session_id, max_pairs=5)
            
            # Refine query with context
            refinement_result = refine_query_with_context(
                user_query, 
                conversation_history, 
                self.config
            )
            
            return (
                refinement_result.is_tennis_related,
                refinement_result.refined_query,
                refinement_result.reasoning
            )
            
        except Exception as e:
            print(f"⚠️ Tennis classification and refinement error: {e}")
            # Default to allowing the query if classifier fails
            return True, user_query, "Classification failed, allowing query"
    
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
            
            # Step 1: Classify and refine query with context
            print(f"\n🎾 Classifying and refining query: '{user_query}'")
            is_tennis_related, refined_query, reasoning = self._classify_and_refine_query(user_query, session_id)
            
            if not is_tennis_related:
                non_tennis_response = "I'm designed specifically to answer tennis-related questions. Please ask me about tennis players, matches, tournaments, rankings, or tennis techniques."
                
                execution_time = time.time() - start_time
                print(f"❌ Non-tennis query detected: {reasoning}")
                
                return {
                    'response': non_tennis_response,
                    'confidence': 1.0,
                    'sources': ['Tennis Classifier'],
                    'action': 'non_tennis_rejection',
                    'processing_time': execution_time
                }
            
            print(f"✅ Tennis-related query confirmed")
            if refined_query != user_query:
                print(f"🔄 Query refined: '{user_query}' → '{refined_query}'")
            
            # Step 2: Orchestrator analyzes query and determines routing
            print(f"\n🧠 Orchestrator analyzing: '{refined_query}'")
            routing_result = self.orchestrator.analyze_and_route(
                refined_query, 
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
                sql_results = self._execute_sql_agent(refined_query, routing_result)
                
                # If SQL fails, fall back to search
                if sql_results and not sql_results.get('success', False):
                    print("⚠️  SQL query failed, falling back to search...")
                    search_results = self._execute_search_agent(refined_query, routing_result)
                    search_needed = True
                
            if search_needed:
                search_results = self._execute_search_agent(refined_query, routing_result)
            
            # Step 4: Synthesize final response
            response = self._synthesize_response(
                refined_query, sql_results, search_results, routing_result
            )
            
            # Store complete conversation in memory
            execution_time = time.time() - start_time
            
            # Store conversation (original query + system response)
            self.memory_manager.store_conversation(
                session_id=session_id,
                user_query=user_query,  # Store original query, not refined
                system_response=response['response']
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
    
    def _execute_sql_agent(self, refined_query: str, routing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL database queries and interpret results efficiently."""
        print("🗄️  Executing SQL Agent...")
        
        try:
            # Generate and execute SQL query
            sql_generation = generate_sql_query.invoke({"user_query": refined_query})
            
            if not sql_generation.get('success', False):
                return {
                    'success': False,
                    'interpretation': f"SQL generation failed: {sql_generation.get('error', 'Unknown error')}",
                    'confidence': 0.0
                }
            
            sql_query = sql_generation.get('sql_query')
            print(f"✨ Generated SQL Query:")
            print(f"   {sql_query}")
            
            # Execute the generated query
            results = execute_sql_query.invoke({"query": sql_query})
            print(f"📊 SQL executed: {results.get('row_count', 0)} rows returned")
            
            # Print the full SQL results
            if results.get('success', False):
                print(f"📋 SQL Results:")
                if results.get('formatted'):
                    print(results['formatted'])
                elif results.get('rows'):
                    print(f"   Columns: {results.get('columns', [])}")
                    for i, row in enumerate(results['rows'][:10]):  # Show first 10 rows
                        print(f"   Row {i+1}: {row}")
                    if len(results['rows']) > 10:
                        print(f"   ... and {len(results['rows']) - 10} more rows")
                else:
                    print("   No data returned")
            else:
                print(f"❌ SQL Error: {results.get('error', 'Unknown error')}")
            
            # Interpret results to natural language
            interpretation = interpret_sql_results.invoke({
                "sql_results": results,
                "user_query": refined_query
            })
            
            return {
                'success': interpretation.get('success', False),
                'interpretation': interpretation.get('interpretation', ''),
                'confidence': interpretation.get('confidence', 0.5),
                'row_count': results.get('row_count', 0),
                'has_data': interpretation.get('has_data', False)
            }
            
        except Exception as e:
            print(f"❌ SQL Agent error: {e}")
            return {
                'success': False,
                'interpretation': f"SQL processing failed: {str(e)}",
                'confidence': 0.0,
                'row_count': 0,
                'has_data': False
            }
    
    def _execute_search_agent(self, refined_query: str, routing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search and interpret results efficiently."""
        print("🌐 Executing Search Agent...")
        
        try:
            # Optimize search query
            optimization_result = optimize_search_query.invoke({
                "user_query": refined_query,
                "context": routing_result.get('routing_decision', {}).get('reasoning', '')
            })
            
            # Use optimized query if successful, fallback to refined if failed
            if optimization_result.get('success', False):
                search_query = optimization_result.get('optimized_query', refined_query)
                print(f"✨ Optimized Search Query:")
                print(f"   {search_query}")
            else:
                search_query = refined_query
                print(f"⚠️  Using Refined Query:")
                print(f"   {refined_query}")
            
            # Perform search
            search_results = tavily_search_tool.invoke({"query": search_query})
            
            result_count = search_results.get('result_count', 0)
            print(f"🔍 Search completed: {result_count} results found")
            
            # Print detailed search results
            raw_results = search_results.get('raw_results', {})
            if raw_results.get('results'):
                print(f"📋 Search Results:")
                for i, result in enumerate(raw_results['results'][:5]):  # Show first 5 results
                    title = result.get('title', 'No title')
                    url = result.get('url', 'No URL')
                    content = result.get('content', 'No content')[:200]  # First 200 chars
                    print(f"   Result {i+1}: {title}")
                    print(f"      URL: {url}")
                    print(f"      Content: {content}{'...' if len(result.get('content', '')) > 200 else ''}")
                    print()
                if len(raw_results['results']) > 5:
                    print(f"   ... and {len(raw_results['results']) - 5} more results")
            else:
                print("   No search results returned")
            
            # Interpret results to natural language
            interpretation = interpret_search_results.invoke({
                "search_results": search_results,
                "user_query": refined_query
            })
            
            return {
                'success': interpretation.get('success', False),
                'interpretation': interpretation.get('interpretation', ''),
                'confidence': interpretation.get('confidence', 0.5),
                'source_count': interpretation.get('source_count', 0),
                'has_data': interpretation.get('has_data', False)
            }
            
        except Exception as e:
            print(f"❌ Search Agent error: {e}")
            return {
                'success': False,
                'interpretation': f"Search processing failed: {str(e)}",
                'confidence': 0.0,
                'source_count': 0,
                'has_data': False
            }
    
    def _synthesize_response(
        self,
        refined_query: str,
        sql_results: Dict[str, Any],
        search_results: Dict[str, Any],
        routing_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize final response from natural language interpretations efficiently."""
        print("🧮 Synthesizing final response...")
        
        # Determine what sources were used
        sql_success = sql_results and sql_results.get('success', False)
        search_success = search_results and search_results.get('success', False)
        
        sources = []
        if sql_success:
            sources.append('Tennis Database')
        if search_success:
            sources.append('Web Search')
        
        # Get current datetime context
        current_datetime = datetime.now()
        current_date_str = current_datetime.strftime("%Y-%m-%d")
        current_year = current_datetime.year
        
        # Build context about steps taken
        steps_taken = []
        sql_interpretation = ""
        search_interpretation = ""
        
        if sql_success:
            sql_interpretation = sql_results.get('interpretation', '')
            row_count = sql_results.get('row_count', 0)
            steps_taken.append(f"✓ Analyzed tennis database ({row_count} records found): {sql_interpretation}")
            print(f"📊 Database analysis: {row_count} records")
        
        if search_success:
            search_interpretation = search_results.get('interpretation', '')
            source_count = search_results.get('source_count', 0)
            steps_taken.append(f"✓ Searched current tennis information ({source_count} sources): {search_interpretation}")
            print(f"🌐 Web search: {source_count} sources")
        
        # Calculate overall confidence
        confidences = []
        if sql_success:
            confidences.append(sql_results.get('confidence', 0.5))
        if search_success:
            confidences.append(search_results.get('confidence', 0.5))
        
        # Handle case where no successful results were obtained
        if not sql_success and not search_success:
            return {
                'response': "I couldn't find specific information about your tennis question. Please try rephrasing your query or asking about a different aspect of tennis.",
                'confidence': 0.1,
                'sources': ['System'],
                'no_data_found': True
            }
        
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.3
        
        try:
            llm = ChatOpenAI(model=self.config.default_model, temperature=0.3)
            
            # Create efficient synthesis prompt
            synthesis_prompt = f"""Create a comprehensive tennis response using the analyzed information.

USER QUESTION: "{refined_query}"

CURRENT CONTEXT: Today is {current_date_str} ({current_year})

ANALYSIS COMPLETED:
{chr(10).join(steps_taken)}

Your task: Combine these insights into a clear, conversational response that directly answers the user's question. Structure your response as follows:

1. **Direct Answer**: Start with a clear answer to their question
2. **Supporting Details**: Include specific facts, numbers, rankings, or statistics 
3. **Context**: Briefly explain how you found this information (database analysis and/or current web search)

Be factual, specific, and engaging. Use the information provided above - don't add information not contained in the analysis results.

IMPORTANT: Keep your response to 100 words or less for memory efficiency.

Response:"""

            response = llm.invoke([
                SystemMessage(content=f"You are a tennis expert providing comprehensive answers using analyzed data. Today is {current_date_str}."),
                HumanMessage(content=synthesis_prompt)
            ])
            
            final_response = response.content.strip()
            
            return {
                'response': final_response,
                'confidence': min(overall_confidence + 0.1, 1.0),  # Slight boost for successful synthesis
                'sources': sources,
                'steps_taken': steps_taken,
                'sql_data_used': sql_success,
                'search_data_used': search_success
            }
            
        except Exception as e:
            print(f"⚠️  LLM synthesis failed: {e}")
            
            # Create fallback response using available interpretations
            fallback_parts = []
            if sql_interpretation:
                fallback_parts.append(f"Database analysis: {sql_interpretation}")
            if search_interpretation:
                fallback_parts.append(f"Current information: {search_interpretation}")
            
            fallback_response = f"Based on the available information:\n\n" + "\n\n".join(fallback_parts) if fallback_parts else "I was unable to find relevant information for your tennis question."
            
            return {
                'response': fallback_response,
                'confidence': overall_confidence * 0.8,  # Reduced confidence for fallback
                'sources': sources,
                'steps_taken': steps_taken,
                'sql_data_used': sql_success,
                'search_data_used': search_success,
                'fallback_used': True
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