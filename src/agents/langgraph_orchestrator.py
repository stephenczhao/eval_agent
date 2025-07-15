"""
Tennis Intelligence System - LangGraph Orchestrator
===================================================

LangGraph-based orchestrator using official tool calling methods with StateGraph,
ToolNode, and proper workflow management for the tennis intelligence system.
"""

import json
import uuid
from typing import Dict, Any, List, Optional, Annotated, TypedDict, Literal
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

try:
    from src.config.settings import TennisConfig
    from src.tools.sql_tools import query_sql_database
    from src.tools.search_tools import online_search
    from src.tools.text_processing_tools import extract_key_entities
except ImportError:
    # Fallback for different import contexts
    from config.settings import TennisConfig
    from tools.sql_tools import query_sql_database
    from tools.search_tools import online_search
    from tools.text_processing_tools import extract_key_entities

import os
import time
import threading
import itertools
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


# Pydantic models for structured output
class ClassificationResult(BaseModel):
    """Tennis query classification result."""
    is_tennis_related: bool = Field(description="Whether the query is tennis-related")
    query_type: Literal["general", "data_specific"] = Field(description="Whether query needs data lookup or can be answered with general knowledge")
    refined_query: str = Field(description="Refined or clarified version of the query")
    reasoning: str = Field(description="Brief explanation of the classification decision")


class RoutingResult(BaseModel):
    """Tennis query routing result."""
    sql_needed: bool = Field(description="Whether SQL database query is needed")
    search_needed: bool = Field(description="Whether web search is needed") 
    reasoning: str = Field(description="Reasoning for routing decision")
    priority: Literal["sql_first", "search_first", "both_parallel"] = Field(description="Execution priority")


class TennisState(TypedDict):
    """State for the tennis intelligence workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    refined_query: str
    query_type: str  # "general" or "data_specific"
    routing_decision: Dict[str, Any]
    sql_results: Optional[Dict[str, Any]]
    search_results: Optional[Dict[str, Any]]
    final_response: Optional[str]
    confidence: float
    sources: List[str]
    error: Optional[str]
    session_id: str


class LangGraphTennisOrchestrator:
    """
    LangGraph-based Tennis Intelligence Orchestrator using official tool calling patterns.
    
    Uses StateGraph to manage workflow with proper tool calling through ToolNode.
    Leverages LangGraph's built-in message-based memory for conversation context.
    """
    
    def __init__(self, config: TennisConfig, debug: bool = False):
        """Initialize the LangGraph orchestrator."""
        self.config = config
        self.debug = debug
        
        # Initialize LLM with tool binding
        self.llm = ChatOpenAI(
            model=config.default_model,
            temperature=config.temperature,
            api_key=config.openai_api_key
        )
        
        # Define available tools
        self.tools = [
            query_sql_database,
            online_search,
            extract_key_entities
        ]
        
        # Bind tools to LLM for proper tool calling
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create tool node for executing tools
        self.tool_node = ToolNode(self.tools)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        # Status indicator setup
        self._status_active = False
        self._status_thread = None
        self._current_status = "thinking..."

    def _debug_print(self, message: str) -> None:
        """Print message only if debug mode is enabled."""
        if self.debug:
            print(message)

    def _start_status_indicator(self, status_message: str = "thinking..."):
        """Start the status indicator with a specific message"""
        if not self.debug and not self._status_active:
            self._current_status = status_message
            self._status_active = True
            self._status_thread = threading.Thread(target=self._run_status_indicator)
            self._status_thread.daemon = True
            self._status_thread.start()

    def _update_status(self, status_message: str):
        """Update the current status message"""
        self._current_status = status_message

    def _stop_status_indicator(self):
        """Stop the status indicator"""
        if self._status_active:
            self._status_active = False
            if self._status_thread:
                self._status_thread.join(timeout=0.1)
            # Clear the line
            if not self.debug:
                print("\r" + " " * 50 + "\r", end="", flush=True)

    def _run_status_indicator(self):
        """Run the status indicator in a separate thread"""
        spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        while self._status_active:
            print(f"\r{next(spinner)} {self._current_status}", end="", flush=True)
            time.sleep(0.1)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for tennis query processing."""
        
        # Create the state graph
        workflow = StateGraph(TennisState)
        
        # Add nodes - simplified workflow since agents handle tools inline
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("router", self._route_query)
        workflow.add_node("sql_agent", self._sql_agent)
        workflow.add_node("search_agent", self._search_agent)
        workflow.add_node("synthesizer", self._synthesize_response)
        
        # Define the workflow flow
        workflow.set_entry_point("classifier")
        
        # Classifier routes conditionally based on query type
        workflow.add_conditional_edges(
            "classifier",
            self._should_route_to_tools,
            {
                "general": "synthesizer",
                "data_specific": "router"
            }
        )
        
        # Router conditionally routes to SQL or search agents
        workflow.add_conditional_edges(
            "router",
            self._should_route_to_sql,
            {
                "sql": "sql_agent",
                "search": "search_agent", 
                "both": "sql_agent"  # Start with SQL if both needed
            }
        )
        
        # SQL agent routes to search or synthesizer
        workflow.add_conditional_edges(
            "sql_agent",
            self._sql_next_step,
            {
                "search_agent": "search_agent",
                "synthesizer": "synthesizer"
            }
        )
        
        # Search agent routes to synthesizer
        workflow.add_edge("search_agent", "synthesizer")
        
        # Synthesizer ends the workflow
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    def _classify_query(self, state: TennisState) -> TennisState:
        """Classify and refine the tennis query using LangGraph's message history for context."""
        self._debug_print(f"🎾 Classifying query: '{state['user_query']}'")
        
        # Create LLM with structured output
        classifier_llm = self.llm.with_structured_output(ClassificationResult)
        
        # Build context from conversation history using LangGraph's message state
        conversation_context = ""
        if state.get("messages"):
            recent_messages = state["messages"][-4:]  # Get last 4 messages for context
            if len(recent_messages) > 1:  # If there's conversation history
                conversation_context = "\n\nCONVERSATION CONTEXT:\n"
                for i, msg in enumerate(recent_messages[:-1]):  # Skip the current query
                    if isinstance(msg, HumanMessage):
                        conversation_context += f"Previous Question: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        conversation_context += f"Previous Answer: {msg.content[:200]}...\n"
        
        # Create classification prompt with conversation context
        classifier_prompt = f"""
        You are an intelligent tennis query classifier. Analyze this query and understand what type of response it needs.
        
        QUERY: "{state['user_query']}"
        {conversation_context}
        
        CONTEXT: This is a tennis intelligence system with access to match data and web search.
        
        CLASSIFICATION LOGIC:
        
        Think about what the user is really asking for:
        
        1. **GENERAL** - Questions about tennis knowledge, techniques, rules, or general guidance:
           - How to play better, technique questions
           - Rules, scoring, equipment advice  
           - Strategy, training, general tennis concepts
           - These can be answered with tennis expertise alone
        
        2. **DATA_SPECIFIC** - Questions that require looking up actual facts, results, or current information:
           - Specific player performance, rankings, statistics
           - Tournament results, match outcomes
           - Current standings, recent events
           - Head-to-head records, career achievements
           - These need real data from databases or current web sources
        
        PRONOUN RESOLUTION: 
        Use conversation context to resolve "he/she/they/him/her" to specific players mentioned earlier.
        Update the query to be clear and specific.
        
        QUERY REFINEMENT:
        If this is a follow-up question, incorporate necessary context from previous conversation.
        Make the refined query self-contained and clear.
        
        Think: Does this need factual data lookup, or can it be answered with general tennis knowledge?
        """
        
        messages = [
            SystemMessage(content="You are a tennis query classifier. Use conversation context to resolve pronouns and clarify queries."),
            HumanMessage(content=classifier_prompt)
        ]
        
        try:
            classification_result = classifier_llm.invoke(messages)
            
            if not classification_result.is_tennis_related:
                state["error"] = f"Query '{state['user_query']}' is not tennis-related"
                state["refined_query"] = state["user_query"]
                self._debug_print(f"❌ Non-tennis query detected: {classification_result.reasoning}")
                return state
            
            state["refined_query"] = classification_result.refined_query
            state["query_type"] = classification_result.query_type
            self._debug_print(f"✅ Tennis query classified as: {classification_result.query_type}")
            self._debug_print(f"🔍 Refined: '{classification_result.refined_query}'")
            if conversation_context:
                self._debug_print(f"🧠 Used conversation context for refinement")
            
        except Exception as e:
            # Fallback: assume tennis-related and data_specific for safety
            state["refined_query"] = state["user_query"]
            state["query_type"] = "data_specific"
            self._debug_print(f"⚠️ Classification fallback used (defaulting to data_specific): {str(e)}")
        
        return state
    
    def _should_route_to_tools(self, state: TennisState) -> str:
        """Determine if query needs tools or can be answered directly."""
        if state.get("error"):
            return "general"  # Route errors to synthesizer for handling
        
        query_type = state.get("query_type", "data_specific")
        self._debug_print(f"🚦 Routing decision: {query_type}")
        
        return query_type
    
    def _route_query(self, state: TennisState) -> TennisState:
        """Determine routing strategy for the query using structured output."""
        if state.get("error"):
            return state
            
        self._debug_print(f"🧠 Routing query: '{state['refined_query']}'")
        
        # Extract entities first
        entity_extraction = extract_key_entities.invoke({"text": state["refined_query"]})
        entities = entity_extraction.get("entities", {})
        
        # Create LLM with structured output
        router_llm = self.llm.with_structured_output(RoutingResult)
        
        # Create routing prompt
        current_date = datetime.now()
        routing_prompt = f"""
        You are a tennis query router. TODAY is {current_date.strftime('%B %d, %Y')}.
        
        QUERY: "{state['refined_query']}"
        ENTITIES: {entities}
        
        DECISION FRAMEWORK:
        
        **WEB SEARCH** - Use when query asks for:
        ✅ Current/live rankings ("who's #1 now", "best player today", "current rankings")
        ✅ Recent events ("last tournament", "latest win", "recent match") 
        ✅ Very recent information or breaking news
        ✅ Current form, live standings
        ✅ "What happened recently?", "Who won the latest tournament?"
        
        **SQL DATABASE** - Use when query asks for:
        ✅ Specific historical matches (2023-2024, early 2025)
        ✅ Career statistics, head-to-head records
        ✅ Tournament history within database range
        ✅ Statistical analysis, win-loss records
        
        **BOTH SOURCES** - Use when query needs:
        ✅ Comparison of historical vs current performance
        ✅ Context that spans both time periods
        
        EXAMPLES:
        - "Who's the best player right now?" → WEB SEARCH (current rankings change frequently)
        - "When did he win his last tournament?" → WEB SEARCH (recent events)
        - "Head to head record between Nadal and Djokovic" → SQL DATABASE (historical stats)
        - "Who won Wimbledon 2024?" → SQL DATABASE (specific historical event)
        
        RULE: If in doubt about recency or currency, choose WEB SEARCH.
        """
        
        messages = [
            SystemMessage(content="You are a tennis query router. Analyze queries to determine optimal data source routing."),
            HumanMessage(content=routing_prompt)
        ]
        
        try:
            routing_result = router_llm.invoke(messages)
            
            # Convert Pydantic model to dict for state storage
            state["routing_decision"] = {
                "sql_needed": routing_result.sql_needed,
                "search_needed": routing_result.search_needed,
                "reasoning": routing_result.reasoning,
                "priority": routing_result.priority
            }
            self._debug_print(f"🎯 Routing: {state['routing_decision']}")
            
        except Exception as e:
            # Simple fallback: default to search for safety when structured output fails
            state["routing_decision"] = {
                "sql_needed": False,
                "search_needed": True,
                "reasoning": f"Fallback to search routing (structured output failed: {str(e)})",
                "priority": "search_first"
            }
            self._debug_print("🔍 Fallback: Defaulting to search when routing analysis fails")
        
        return state
    
    def _sql_agent(self, state: TennisState) -> TennisState:
        """SQL agent that uses tool calling for database operations."""
        if state.get("error"):
            return state
            
        self._debug_print("🗄️ SQL Agent executing...")
        self._update_status("querying SQL database...")
        
        # Create a conversation with tool calling support
        conversation_messages = []
        
        # Add system message
        conversation_messages.append(
            SystemMessage(content="You are a tennis SQL agent. Use the query_sql_database tool to answer tennis questions.")
        )
        
        # Add the user query
        conversation_messages.append(
            HumanMessage(content=f"Answer this tennis query using the SQL database: {state['refined_query']}")
        )
        
        # Keep calling LLM until it stops making tool calls
        max_iterations = 3
        for iteration in range(max_iterations):
            # Get LLM response (may include tool calls)
            response = self.llm_with_tools.invoke(conversation_messages)
            conversation_messages.append(response)
            state["messages"].append(response)
            
            # If no tool calls, we're done
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                break
                
            # Debug: Print tool calls that were made
            self._debug_print(f"🔧 SQL Agent called {len(response.tool_calls)} tools:")
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                self._debug_print(f"   📞 Calling: {tool_name}")
            
            # Execute all tool calls at once (ToolNode processes all tool calls together)
                try:
                # Execute all tools from this response at once
                    tool_result = self.tool_node.invoke({"messages": [response]})
                    
                # Add all tool messages to conversation
                    if tool_result and "messages" in tool_result:
                        for tool_message in tool_result["messages"]:
                            conversation_messages.append(tool_message)
                            state["messages"].append(tool_message)
                            
                except Exception as e:
                    self._debug_print(f"❌ Tool execution failed: {e}")
                # Create error tool messages for each tool call
                    from langchain_core.messages import ToolMessage
                for tool_call in response.tool_calls:
                    error_message = ToolMessage(
                        content=f"Tool execution failed: {str(e)}",
                        tool_call_id=tool_call["id"]
                    )
                    conversation_messages.append(error_message)
                    state["messages"].append(error_message)
        
        return state
    
    def _search_agent(self, state: TennisState) -> TennisState:
        """Search agent that uses tool calling for web search."""
        if state.get("error"):
            return state
            
        self._debug_print("🌐 Search Agent executing...")
        self._update_status("searching online...")
        
        # Create a conversation with tool calling support
        conversation_messages = []
        
        # Add system message
        conversation_messages.append(
            SystemMessage(content="You are a tennis search agent. Use the online_search tool to get current tennis information.")
        )
        
        # Add the user query
        conversation_messages.append(
            HumanMessage(content=f"Answer this tennis query using online search: {state['refined_query']}")
        )
        
        # Keep calling LLM until it stops making tool calls
        max_iterations = 3
        for iteration in range(max_iterations):
            # Get LLM response (may include tool calls)
            response = self.llm_with_tools.invoke(conversation_messages)
            conversation_messages.append(response)
            state["messages"].append(response)
            
            # If no tool calls, we're done
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                break
                
            # Debug: Print tool calls that were made
            self._debug_print(f"🔧 Search Agent called {len(response.tool_calls)} tools:")
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                self._debug_print(f"   📞 Calling: {tool_name}")
            
            # Execute all tool calls at once (ToolNode processes all tool calls together)
                try:
                # Execute all tools from this response at once
                    tool_result = self.tool_node.invoke({"messages": [response]})
                    
                # Add all tool messages to conversation
                    if tool_result and "messages" in tool_result:
                        for tool_message in tool_result["messages"]:
                            conversation_messages.append(tool_message)
                            state["messages"].append(tool_message)
                            
                except Exception as e:
                    self._debug_print(f"❌ Tool execution failed: {e}")
                # Create error tool messages for each tool call
                    from langchain_core.messages import ToolMessage
                for tool_call in response.tool_calls:
                    error_message = ToolMessage(
                        content=f"Tool execution failed: {str(e)}",
                        tool_call_id=tool_call["id"]
                    )
                    conversation_messages.append(error_message)
                    state["messages"].append(error_message)
        
        return state
    
    def _synthesize_response(self, state: TennisState) -> TennisState:
        """Synthesize final response from collected data or general knowledge."""
        if state.get("error"):
            return state
            
        self._debug_print("🧮 Synthesizing final response...")
        self._update_status("preparing final answer...")
        
        query_type = state.get("query_type", "data_specific")
        
        # Handle general queries that don't need tools
        if query_type == "general":
            return self._synthesize_general_response(state)
        
        # Handle data-specific queries with tool results
        self._debug_print_tool_results(state)
        
        # Collect all information from the conversation
        all_info = []
        sources = []
        
        for message in state["messages"]:
            if hasattr(message, 'content') and message.content:
                all_info.append(str(message.content))
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if 'query_sql_database' in tool_call.get('name', ''):
                        sources.append("Tennis Database")
                    elif 'online_search' in tool_call.get('name', ''):
                        sources.append("Web Search")
        
        # Create synthesis prompt with conversation history for context
        synthesis_prompt = f"""
        Create a comprehensive tennis response for this query:
        
        QUERY: "{state['refined_query']}"
        
        AVAILABLE INFORMATION:
        {chr(10).join(all_info[-5:])}  # Last 5 pieces of info
        
        SOURCES USED: {list(set(sources))}
        
        Provide a clear, factual response that directly answers the query.
        
        LENGTH: Keep your response to 100 words or less. Be concise and focused.
        
        Use the conversation context naturally - LangGraph will handle memory automatically.
        """
        
        # Include conversation context in the messages for natural memory handling
        synthesis_messages = []
        
        # Add system message
        synthesis_messages.append(
            SystemMessage(content=f"You are a tennis expert. Today is {datetime.now().strftime('%Y-%m-%d')}.")
        )
        
        # Add recent conversation context for natural memory, but be careful with tool calls
        if state.get("messages"):
            # Include last few exchanges but handle tool calls properly
            recent_messages = state["messages"][-10:]  # Get more messages for context
            
            # Filter and organize messages to ensure proper tool call sequences
            filtered_messages = []
            i = 0
            while i < len(recent_messages):
                msg = recent_messages[i]
                
                if isinstance(msg, HumanMessage):
                    # Always include human messages
                    filtered_messages.append(msg)
                    i += 1
                elif isinstance(msg, AIMessage):
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        # This AI message has tool calls - include it with all its tool responses
                        filtered_messages.append(msg)
                        i += 1
                        
                        # Find and include all corresponding tool messages
                        while i < len(recent_messages) and isinstance(recent_messages[i], ToolMessage):
                            filtered_messages.append(recent_messages[i])
                            i += 1
                    else:
                        # Regular AI message without tool calls
                        filtered_messages.append(msg)
                        i += 1
                else:
                    # Skip standalone tool messages or other message types
                    i += 1
            
            # Add the filtered messages, keeping only the most recent ones
            synthesis_messages.extend(filtered_messages[-6:])  # Last 6 filtered messages
        
        # Add the synthesis prompt
        synthesis_messages.append(HumanMessage(content=synthesis_prompt))
        
        response = self.llm.invoke(synthesis_messages)
        
        state["final_response"] = response.content
        state["sources"] = list(set(sources)) if sources else ["System"]
        state["confidence"] = 0.8 if sources else 0.5
        
        # Add the response to messages for LangGraph's natural memory handling
        state["messages"].append(response)
        
        self._debug_print(f"✅ Response synthesized using LangGraph's built-in memory")
        
        return state
    
    def _synthesize_general_response(self, state: TennisState) -> TennisState:
        """Synthesize response for general tennis questions using LLM knowledge."""
        self._debug_print("🎓 Answering general tennis question with knowledge base")
        
        # Create prompt for general tennis knowledge
        general_prompt = f"""
        Answer this tennis question using your general tennis knowledge:
        
        QUERY: "{state['refined_query']}"
        
        You are a tennis expert. Provide a clear, informative, and helpful response.
        Include practical tips, explanations, or guidance as appropriate.
        
        LENGTH: Keep your response to 100 words or less. Be concise and focused.
        
        Since this is a general tennis question, you don't need to cite specific recent data or statistics.
        Focus on providing accurate tennis knowledge, techniques, rules, or general guidance.
        """
        
        # Create messages for LLM
        general_messages = [
            SystemMessage(content="You are a knowledgeable tennis expert providing helpful guidance on tennis techniques, rules, and general knowledge."),
            HumanMessage(content=general_prompt)
        ]
        
        try:
            response = self.llm.invoke(general_messages)
            
            # Store the response
            state["final_response"] = response.content
            state["sources"] = ["Tennis Knowledge Base"]
            state["confidence"] = 0.9  # High confidence for general knowledge
            
            # Add the response to messages for LangGraph's memory handling
            state["messages"].append(response)
            
            self._debug_print(f"✅ General tennis response synthesized")
            
        except Exception as e:
            error_msg = f"Failed to generate general tennis response: {str(e)}"
            state["final_response"] = error_msg
            state["sources"] = ["System"]
            state["confidence"] = 0.0
            state["error"] = error_msg
            self._debug_print(f"❌ {error_msg}")
            
        return state
    
    def _debug_print_tool_results(self, state: TennisState) -> None:
        """Debug helper to print tool results in a formatted way"""
        if not self.debug or not state.get("messages"):
            return
            
        self._debug_print("\n📋 DEBUG: Tool Results Summary")
        self._debug_print("=" * 50)
        
        for i, message in enumerate(state["messages"]):
            self._debug_print(f"\nMessage {i+1}: {type(message).__name__}")
            
            # Check for tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    self._debug_print(f"🔧 Tool Call: {tool_name}")
                    
                    # Print tool arguments
                    if 'args' in tool_call:
                        args = tool_call['args']
                        if tool_name == 'generate_sql_query' and 'user_query' in args:
                            self._debug_print(f"   🎯 Query: {args['user_query']}")
                        elif tool_name == 'execute_sql_query' and 'query' in args:
                            self._debug_print(f"   ✨ Generated SQL:")
                            self._debug_print(f"      {args['query']}")
                        else:
                            self._debug_print(f"   📝 Args: {args}")
            
            # Check for all message content
            if hasattr(message, 'content') and message.content:
                content = str(message.content)
                self._debug_print(f"📄 Content: {content[:150]}{'...' if len(content) > 150 else ''}")
                
                # Try to parse specific tool results
                if 'query_sql_database' in content.lower() or ('generated_sql' in content.lower() and 'interpretation' in content.lower()):
                    self._debug_print(f"   🎯 SQL Database Query Result:")
                    try:
                        import json
                        result = json.loads(content)
                        if result.get('success'):
                            if 'generated_sql' in result:
                                self._debug_print(f"      ✨ Generated SQL: {result['generated_sql']}")
                            if 'row_count' in result:
                                self._debug_print(f"      📊 Rows returned: {result['row_count']}")
                            if 'interpretation' in result:
                                self._debug_print(f"      💬 Interpretation: {result['interpretation'][:200]}{'...' if len(result['interpretation']) > 200 else ''}")
                            if 'tools_called' in result:
                                self._debug_print(f"      🔧 Tools used: {result['tools_called']}")
                        else:
                            self._debug_print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
                    except:
                        self._debug_print(f"      {content[:300]}{'...' if len(content) > 300 else ''}")
                
                elif 'online_search' in content.lower() or ('optimized_query' in content.lower() and 'interpretation' in content.lower()):
                    self._debug_print(f"   🌐 Online Search Result:")
                    try:
                        import json
                        result = json.loads(content)
                        if result.get('success'):
                            if 'optimized_query' in result:
                                self._debug_print(f"      🔍 Optimized query: {result['optimized_query']}")
                            if 'result_count' in result:
                                self._debug_print(f"      📊 Results found: {result['result_count']}")
                            if 'interpretation' in result:
                                self._debug_print(f"      💬 Interpretation: {result['interpretation'][:200]}{'...' if len(result['interpretation']) > 200 else ''}")
                            if 'tools_called' in result:
                                self._debug_print(f"      🔧 Tools used: {result['tools_called']}")
                        else:
                            self._debug_print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
                    except:
                        self._debug_print(f"      {content[:300]}{'...' if len(content) > 300 else ''}")
                
                elif 'interpretation' in content.lower():
                    self._debug_print(f"   🎾 Other Tool Result:")
                    try:
                        import json
                        result = json.loads(content)
                        if 'interpretation' in result:
                            self._debug_print(f"      💬 {result['interpretation'][:200]}{'...' if len(result['interpretation']) > 200 else ''}")
                    except:
                        self._debug_print(f"      {content[:200]}{'...' if len(content) > 200 else ''}")
        
        self._debug_print("=" * 50)
    
    # Conditional routing functions
    def _should_route_to_sql(self, state: TennisState) -> str:
        """Determine initial routing based on routing decision."""
        if state.get("error"):
            return "search"  # Route to search as fallback, not synthesize
            
        routing = state.get("routing_decision", {})
        sql_needed = routing.get("sql_needed", False)
        search_needed = routing.get("search_needed", False)
        
        if sql_needed and search_needed:
            return "both"
        elif sql_needed:
            return "sql"
        elif search_needed:
            return "search"
        else:
            return "sql"  # Default
    
    def _sql_next_step(self, state: TennisState) -> str:
        """Determine next step after SQL agent."""
        # Check if we need search too based on routing
        routing = state.get("routing_decision", {})
        if routing.get("search_needed", False):
            return "search_agent"
        
        # Otherwise go straight to synthesis
        return "synthesizer"
    
    def process_query(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """Process a tennis query through the LangGraph workflow."""
        start_time = time.time()
        
        # Start status indicator
        self._start_status_indicator("thinking...")
        
        try:
            self._debug_print(f"\n🚀 LangGraph Processing: '{user_query}'")
            
            # Initialize state with user query as a message for LangGraph memory
            initial_state = TennisState(
                messages=[HumanMessage(content=user_query)],
                user_query=user_query,
                refined_query="",
                query_type="data_specific",  # Default value, will be set by classifier
                routing_decision={},
                sql_results=None,
                search_results=None,
                final_response=None,
                confidence=0.0,
                sources=[],
                error=None,
                session_id=session_id
            )
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Extract results
            response = final_state.get("final_response", "I couldn't process your tennis query.")
            confidence = final_state.get("confidence", 0.0)
            sources = final_state.get("sources", ["System"])
            error = final_state.get("error")
            
            # Extract SQL query from tool results for Text2SQL evaluation
            sql_query = self._extract_sql_query_from_messages(final_state.get("messages", []))
            
            processing_time = time.time() - start_time
            
            result = {
                'response': response,
                'confidence': confidence,
                'sources': sources,
                'sql_data_used': 'Tennis Database' in sources,
                'search_data_used': 'Web Search' in sources,
                'error': error is not None,
                'langgraph_used': True,
                'processing_time': processing_time
            }
            
            # Add SQL query if found (for Text2SQL evaluation)
            if sql_query:
                result['sql_query'] = sql_query
                self._debug_print(f"🔍 Extracted SQL for evaluation: {sql_query}")
            
            return result
            
        except Exception as e:
            error_msg = f"LangGraph workflow error: {str(e)}"
            self._debug_print(f"❌ {error_msg}")
            return {
                'response': error_msg,
                'confidence': 0.0,
                'sources': ['System'],
                'error': True,
                'langgraph_used': True,
                'processing_time': time.time() - start_time
            }
        finally:
            # Stop status indicator
            self._stop_status_indicator() 
    
    def _extract_sql_query_from_messages(self, messages: List[BaseMessage]) -> Optional[str]:
        """Extract the SQL query from tool messages for Text2SQL evaluation."""
        try:
            for message in messages:
                if hasattr(message, 'content') and message.content:
                    content = str(message.content)
                    
                    # Look for SQL database tool results
                    if 'generated_sql' in content.lower() or 'query_sql_database' in content.lower():
                        try:
                            import json
                            result = json.loads(content)
                            
                            # Extract SQL query from different possible locations
                            if result.get('generated_sql'):
                                return result['generated_sql']
                            elif result.get('sql_query'):
                                return result['sql_query']
                                
                        except json.JSONDecodeError:
                            # Try to extract SQL with regex if JSON parsing fails
                            import re
                            sql_pattern = r'Generated SQL:\s*(.+?)(?:\n|$|,)'
                            match = re.search(sql_pattern, content, re.IGNORECASE)
                            if match:
                                return match.group(1).strip()
                            
                            # Try another pattern
                            sql_pattern = r'"generated_sql":\s*"([^"]+)"'
                            match = re.search(sql_pattern, content, re.IGNORECASE)
                            if match:
                                return match.group(1).strip()
            
            return None
            
        except Exception as e:
            self._debug_print(f"❌ Error extracting SQL query: {e}")
            return None 