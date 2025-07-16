"""
Tennis Intelligence System - LangGraph Orchestrator
===================================================

LangGraph-based orchestrator using official tool calling methods with StateGraph,
ToolNode, and proper workflow management for the tennis intelligence system.
"""

import json
import uuid
from typing import Dict, Any, List, Optional, Annotated, TypedDict, Literal, Union
from datetime import datetime
from pydantic import BaseModel, Field
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# Fallback for different import contexts
from ..config.settings import TennisConfig
from ..tools.sql_tools import query_sql_database, execute_sql_query, generate_sql_query, interpret_sql_results
from ..tools.search_tools import online_search, tavily_search_tool, interpret_search_results
from ..tools.text_processing_tools import extract_key_entities

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


class QueryTypeDecision(BaseModel):
    """Decision for query type routing."""
    route_to: Literal["general", "data_specific"] = Field(description="Where to route the query")
    reasoning: str = Field(description="Reasoning for the routing decision")


class ToolRoutingDecision(BaseModel):
    """Decision for tool routing."""
    route_to: Literal["sql", "search", "both"] = Field(description="Which tools to use")
    reasoning: str = Field(description="Reasoning for the tool routing decision")


class NextStepDecision(BaseModel):
    """Decision for next step after tool execution."""
    next_step: Literal["search_agent", "synthesizer"] = Field(description="Next step in the workflow")
    reasoning: str = Field(description="Reasoning for the next step decision")


class TennisState(TypedDict):
    """State for the tennis intelligence workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    refined_query: str
    query_type: str  # "general" or "data_specific"
    routing_decision: Dict[str, Any]
    query_type_decision: Dict[str, Any]
    tool_routing_decision: Dict[str, Any]
    next_step_decision: Dict[str, Any]
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
        
        # Simple JSON-based memory for CLI sessions
        self.memory_file = Path("tennis_data/session_memory.json")
        self.memory_file.parent.mkdir(exist_ok=True)
        
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

    def _load_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Load conversation history from JSON file."""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    all_sessions = json.load(f)
                return all_sessions.get(session_id, [])
            return []
        except Exception as e:
            if self.debug:
                print(f"⚠️ Failed to load memory: {e}")
            return []

    def _save_conversation(self, session_id: str, user_query: str, final_answer: str) -> None:
        """Save conversation to JSON file."""
        try:
            # Load existing conversations
            all_sessions = {}
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    all_sessions = json.load(f)
            
            # Initialize session if it doesn't exist
            if session_id not in all_sessions:
                all_sessions[session_id] = []
            
            # Add new conversation
            conversation_id = len(all_sessions[session_id])
            all_sessions[session_id].append({
                "conversation_id": conversation_id,
                "user_query": user_query,
                "final_answer": final_answer,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save back to file
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(all_sessions, f, indent=2, ensure_ascii=False)
                
            if self.debug:
                print(f"💾 Saved conversation {conversation_id} to memory")
                
        except Exception as e:
            if self.debug:
                print(f"⚠️ Failed to save memory: {e}")

    def _build_conversation_context(self, session_id: str) -> List[BaseMessage]:
        """Build conversation context from memory for LangGraph state."""
        history = self._load_conversation_history(session_id)
        messages = []
        
        # Add recent conversations (last 3 pairs to avoid overwhelming context)
        recent_history = history[-3:] if len(history) > 3 else history
        
        for conv in recent_history:
            messages.append(HumanMessage(content=conv["user_query"]))
            messages.append(AIMessage(content=conv["final_answer"]))
        
        if self.debug and recent_history:
            print(f"🧠 Loaded {len(recent_history)} previous conversations for context")
            
        return messages

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
                print("\r" + " " * 100 + "\r", end="", flush=True)

    def _run_status_indicator(self):
        """Run the status indicator in a separate thread"""
        spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        while self._status_active:
            # Clear the line and print new status
            status_line = f"{next(spinner)} {self._current_status}"
            print(f"\r{' ' * 100}\r{status_line}", end="", flush=True)
            time.sleep(0.1)
    
    def _build_workflow(self):
        """Build the LangGraph workflow for tennis query processing."""
        
        # Create the state graph
        workflow = StateGraph(TennisState)
        
        # Add nodes - simplified workflow since agents handle tools inline
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("query_type_router", self._route_by_query_type)
        workflow.add_node("tool_router", self._route_by_tools)
        workflow.add_node("sql_agent", self._sql_agent)
        workflow.add_node("search_agent", self._search_agent)
        workflow.add_node("next_step_router", self._route_next_step)
        workflow.add_node("synthesizer", self._synthesize_response)
        
        # Define the workflow flow
        workflow.set_entry_point("classifier")
        
        # Classifier always goes to query type router
        workflow.add_edge("classifier", "query_type_router")
        
        # Query type router conditionally routes based on LLM decision
        workflow.add_conditional_edges(
            "query_type_router",
            self._get_query_type_route,
            {
                "general": "synthesizer",
                "data_specific": "tool_router"
            }
        )
        
        # Tool router conditionally routes to SQL or search agents based on LLM decision
        workflow.add_conditional_edges(
            "tool_router",
            self._get_tool_route,
            {
                "sql": "sql_agent",
                "search": "search_agent", 
                "both": "sql_agent"  # Start with SQL if both needed
            }
        )
        
        # SQL agent routes to next step router
        workflow.add_edge("sql_agent", "next_step_router")
        
        # Next step router conditionally routes based on LLM decision
        workflow.add_conditional_edges(
            "next_step_router",
            self._get_next_step_route,
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
            recent_messages = state["messages"][-7:]  # Get last 7 messages (3 conversation pairs + current)
            if len(recent_messages) > 1:  # If there's conversation history
                conversation_context = "\n\nRECENT CONVERSATION CONTEXT:\n"
                
                # Process message pairs (skip the current query which is last)
                for i in range(0, len(recent_messages) - 1, 2):  # Step by 2 to get pairs
                    if i + 1 < len(recent_messages) - 1:  # Make sure we have both question and answer
                        human_msg = recent_messages[i]
                        ai_msg = recent_messages[i + 1]
                        
                        if isinstance(human_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                            conversation_context += f"Previous Question: {human_msg.content}\n"
                            conversation_context += f"Previous Answer: {ai_msg.content[:150]}...\n\n"
        
        # Get current datetime context
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year
        current_month = datetime.now().strftime("%B %Y")
        
        # Create classification prompt with conversation context
        classifier_prompt = f"""
        You are an intelligent tennis query classifier. Analyze this query and understand what type of response it needs.
        
        QUERY: "{state['user_query']}"
        {conversation_context}
        
        CONTEXT: This is a tennis intelligence system with access to match data and web search.
        
        CURRENT DATETIME CONTEXT:
        - Today's date: {current_date}
        - Current year: {current_year}
        - Current month: {current_month}
        - Database contains tennis matches from 2023-01-01 to 2025-06-28
        - disregard your training cutoff time.
        
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
        
        CRITICAL - PRONOUN RESOLUTION AND CONTEXT AWARENESS: 
        - If the query contains pronouns ("he/she/they/him/her"), resolve them using the conversation context
        - If the query is incomplete ("what about...", "her win ratio", "how many for that year"), incorporate context
        - Make the refined query self-contained by including the specific player name and time period from context
        - Example: "what was her win ratio for that year?" + context about "Sabalenka" and "2024" → "What was Sabalenka's win ratio for 2024?"
        
        User asked the question at {datetime.now().strftime("%m/%d/%Y")}, disregard your training cutoff time.
        QUERY REFINEMENT:
        Always make the refined query complete and specific, using conversation context AND datetime context to fill in missing information.
        
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
    
    def _route_by_query_type(self, state: TennisState) -> TennisState:
        """Route based on query type using LLM decision."""
        if state.get("error"):
            state["query_type_decision"] = {"route_to": "general", "reasoning": "Error state"}
            return state
            
        self._debug_print(f"🚦 Routing by query type: '{state['refined_query']}'")
        
        # Create LLM with structured output for routing decision
        router_llm = self.llm.with_structured_output(QueryTypeDecision)
        
        routing_prompt = f"""
        You are a tennis query router. Determine if this query should be handled as a general knowledge question or requires specific data lookup.
        
        QUERY: "{state['refined_query']}"
        CURRENT CLASSIFICATION: {state.get('query_type', 'unknown')}
        
        ROUTING OPTIONS:
        - "general": Route to general tennis knowledge (no tools needed)
        - "data_specific": Route to tool-based data lookup
        
        Consider:
        - Does this need current/historical tennis data?
        - Can this be answered with general tennis knowledge?
        - Is this asking for specific facts, statistics, or results?
        
        Choose the most appropriate route based on the query requirements.
        """
        
        messages = [
            SystemMessage(content="You are a tennis query router. Make routing decisions based on query requirements."),
            HumanMessage(content=routing_prompt)
        ]
        
        try:
            routing_result = router_llm.invoke(messages)
            route_to = getattr(routing_result, 'route_to', 'data_specific')
            state["query_type_decision"] = {
                "route_to": route_to,
                "reasoning": getattr(routing_result, 'reasoning', 'No reasoning provided')
            }
            self._debug_print(f"🎯 Route: {route_to}")
            
        except Exception as e:
            # Fallback to data_specific for safety
            state["query_type_decision"] = {
                "route_to": "data_specific",
                "reasoning": f"Fallback due to routing error: {str(e)}"
            }
            self._debug_print(f"⚠️ Query type routing fallback: {str(e)}")
        
        return state
    
    def _route_by_tools(self, state: TennisState) -> TennisState:
        """Route to appropriate tools using LLM decision."""
        if state.get("error"):
            state["tool_routing_decision"] = {"route_to": "search", "reasoning": "Error state fallback"}
            return state
            
        self._debug_print(f"🧠 Routing by tools: '{state['refined_query']}'")
        
        # Extract entities first
        entity_extraction = extract_key_entities.invoke({"text": state["refined_query"]})
        entities = entity_extraction.get("entities", {})
        
        # Create LLM with structured output
        router_llm = self.llm.with_structured_output(ToolRoutingDecision)
        
        # Create routing prompt
        current_date = datetime.now()
        routing_prompt = f"""
        You are a tennis tool router. TODAY is {current_date.strftime('%B %d, %Y')}.
        
        QUERY: "{state['refined_query']}"
        ENTITIES: {entities}
        
        ROUTING OPTIONS:
        - "sql": Use SQL database for historical data from 2023-01-01 to 2025-06-28 (DO NOT CALL THIS DATABASE FOR INFORMATION OUTSIDE THIS RANGE)
        - "search": Use web search for current/recent information or information outside the database range
        - "both": Use both tools for comprehensive coverage
        
        DECISION FRAMEWORK:
        
        **WEB SEARCH** - Choose when query asks for:
        ✅ Current/live rankings ("who's #1 now", "best player today", "current rankings")
        ✅ Recent events ("last tournament", "latest win", "recent match") 
        ✅ Very recent information or breaking news
        ✅ Current form, live standings
        ✅ "What happened recently?", "Who won the latest tournament?"
        
        **SQL DATABASE** - Choose when query asks for:
        ✅ Specific historical matches (2023-2024, early 2025)
        ✅ Career statistics, head-to-head records
        ✅ Tournament history within database range
        ✅ Statistical analysis, win-loss records
        
        **BOTH TOOLS** - Choose when query needs:
        ✅ Comparison of historical vs current performance
        ✅ Context that spans both time periods
        
        EXAMPLES:
        - "Who's the best player right now?" → search (current rankings change frequently)
        - "When did he win his last tournament?" → search (recent events)
        - "Head to head record between Nadal and Djokovic" → sql (historical stats)
        - "Who won Wimbledon 2024?" → sql (specific historical event)
        
        RULE: If in doubt about recency or currency, choose search.
        
        Choose the most appropriate tool routing option.
        """
        
        messages = [
            SystemMessage(content="You are a tennis tool router. Analyze queries to determine optimal tool routing."),
            HumanMessage(content=routing_prompt)
        ]
        
        try:
            routing_result = router_llm.invoke(messages)
            route_to = getattr(routing_result, 'route_to', 'search')
            state["tool_routing_decision"] = {
                "route_to": route_to,
                "reasoning": getattr(routing_result, 'reasoning', 'No reasoning provided')
            }
            self._debug_print(f"🎯 Tool: {route_to}")
            
        except Exception as e:
            # Simple fallback: default to search for safety
            state["tool_routing_decision"] = {
                "route_to": "search",
                "reasoning": f"Fallback to search routing: {str(e)}"
            }
            self._debug_print(f"🔍 Tool routing fallback: {str(e)}")
        
        return state
    
    def _route_next_step(self, state: TennisState) -> TennisState:
        """Determine next step after SQL execution using LLM decision."""
        if state.get("error"):
            state["next_step_decision"] = {"next_step": "synthesizer", "reasoning": "Error state"}
            return state
            
        self._debug_print(f"🔄 Routing next step after SQL")
        
        # Create LLM with structured output
        router_llm = self.llm.with_structured_output(NextStepDecision)
        
        # Analyze the conversation and tool routing to determine if search is needed
        tool_routing = state.get("tool_routing_decision", {})
        recent_messages = state.get("messages", [])[-3:]  # Get recent messages for context
        
        routing_prompt = f"""
        You are a workflow router. Determine the next step after SQL execution.
        
        ORIGINAL QUERY: "{state['refined_query']}"
        TOOL ROUTING DECISION: {tool_routing}
        
        NEXT STEP OPTIONS:
        - "search_agent": Continue to web search for additional information
        - "synthesizer": Go directly to synthesize final response
        
        DECISION LOGIC:
        - Choose "search_agent" if:
          * Original tool routing indicated "both" tools needed
          * SQL results might be incomplete or need current context
          * Query asks for both historical and recent information
        
        - Choose "synthesizer" if:
          * Only SQL was needed according to tool routing
          * SQL results appear sufficient to answer the query
          * No additional web search context required
        
        Based on the tool routing decision and query requirements, what should be the next step?
        """
        
        messages = [
            SystemMessage(content="You are a workflow router. Determine optimal next steps based on tool routing decisions."),
            HumanMessage(content=routing_prompt)
        ]
        
        try:
            routing_result = router_llm.invoke(messages)
            next_step = getattr(routing_result, 'next_step', 'synthesizer')
            state["next_step_decision"] = {
                "next_step": next_step,
                "reasoning": getattr(routing_result, 'reasoning', 'No reasoning provided')
            }
            self._debug_print(f"🔄 Next: {next_step}")
            
        except Exception as e:
            # Default to synthesizer for safety
            state["next_step_decision"] = {
                "next_step": "synthesizer",
                "reasoning": f"Fallback to synthesizer: {str(e)}"
            }
            self._debug_print(f"⚠️ Next step fallback: {str(e)}")
        
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
            
            # Check if response is AIMessage with tool calls
            if isinstance(response, AIMessage) and hasattr(response, 'tool_calls') and response.tool_calls:
                # Debug: Print tool calls that were made
                tool_names = [tool_call.get('name', 'unknown') for tool_call in response.tool_calls]
                self._debug_print(f"🔧 Calling: {', '.join(tool_names)}")
                
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
            else:
                # No tool calls, we're done
                break
        
        return state
    
    def _search_agent(self, state: TennisState) -> TennisState:
        """Search agent that uses tool calling for web search."""
        if state.get("error"):
            return state
            
        self._debug_print("🌐 Search Agent executing...")
        # Show the actual query being searched for
        self._update_status(f"searching online for: {state['refined_query']}")
        
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
            
            # Check if response is AIMessage with tool calls
            if isinstance(response, AIMessage) and hasattr(response, 'tool_calls') and response.tool_calls:
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
            else:
                # No tool calls, we're done
                break
        
        return state
    
    def _synthesize_response(self, state: TennisState) -> TennisState:
        """Synthesize final response from collected data or general knowledge."""
        if state.get("error"):
            return state
            
        self._debug_print("🧮 Synthesizing final response...")
        self._update_status("preparing final answer...")
        
        query_type_decision = state.get("query_type_decision", {})
        route_to = query_type_decision.get("route_to", "data_specific")
        
        # Handle general queries that don't need tools
        if route_to == "general":
            return self._synthesize_general_response(state)
        
        # Handle data-specific queries with tool results
        # Note: Tool results already shown during execution
        
        # Collect all information from the conversation
        all_info = []
        sources = []
        
        for message in state["messages"]:
            if hasattr(message, 'content') and message.content:
                all_info.append(str(message.content))
            if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
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
        
        # Handle both string and list content types
        response_content = ""
        if isinstance(response.content, str):
            response_content = response.content
        elif isinstance(response.content, list):
            # If content is a list, join string parts
            response_content = " ".join([str(part) for part in response.content if isinstance(part, str)])
        
        state["final_response"] = response_content
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
        
        response = self.llm.invoke(general_messages)
        
        # Handle both string and list content types
        response_content = ""
        if isinstance(response.content, str):
            response_content = response.content
        elif isinstance(response.content, list):
            # If content is a list, join string parts
            response_content = " ".join([str(part) for part in response.content if isinstance(part, str)])
        
        # Store the response
        state["final_response"] = response_content
        state["sources"] = ["Tennis Knowledge Base"]
        state["confidence"] = 0.9  # High confidence for general knowledge
        
        # Add the response to messages for LangGraph's memory handling
        state["messages"].append(response)
        
        self._debug_print(f"✅ General tennis response synthesized")
            
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
            if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
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
                
                # Check if this is a SQL query result - show full details in debug mode
                if 'query_sql_database' in content.lower() or ('sql_query' in content.lower() and 'success' in content.lower()):
                    self._debug_print(f"📄 Content: SQL Database Result")
                    self._debug_print(f"   🎯 SQL Database Query Result:")
                    try:
                        import json
                        result = json.loads(content)
                        if result.get('success'):
                            # Show full SQL query
                            if 'sql_query' in result:
                                self._debug_print(f"      📝 Full SQL Query:")
                                self._debug_print(f"         {result['sql_query']}")
                            
                            # Show query results details
                            if 'raw_results' in result:
                                raw_results = result['raw_results']
                                self._debug_print(f"      📊 Raw Query Results:")
                                if isinstance(raw_results, list) and raw_results:
                                    self._debug_print(f"         Rows returned: {len(raw_results)}")
                                    # Show first 10 rows of actual data
                                    for i, row in enumerate(raw_results[:10]):  # Show first 10 rows
                                        self._debug_print(f"         Row {i+1}: {row}")
                                    if len(raw_results) > 10:
                                        self._debug_print(f"         ... and {len(raw_results) - 10} more rows")
                                elif raw_results is not None:
                                    self._debug_print(f"         {raw_results}")
                                else:
                                    self._debug_print(f"         No data returned")
                            
                            if 'row_count' in result:
                                self._debug_print(f"      📈 Total rows: {result['row_count']}")
                            if 'interpretation' in result:
                                self._debug_print(f"      💬 Full Interpretation:")
                                self._debug_print(f"         {result['interpretation']}")
                        else:
                            self._debug_print(f"      ❌ SQL Query Failed: {result.get('error', 'Unknown error')}")
                            if 'sql_query' in result:
                                self._debug_print(f"      📝 Failed SQL Query: {result['sql_query']}")
                    except Exception as e:
                        self._debug_print(f"      ⚠️ Could not parse SQL result: {e}")
                        self._debug_print(f"      📄 Raw content: {content}")
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
                else:
                    # For non-SQL, non-search content, show truncated version
                    self._debug_print(f"📄 Content: {content[:150]}{'...' if len(content) > 150 else ''}")
        
        self._debug_print("=" * 50)
    
    # Conditional routing functions - now prompt-based
    def _get_query_type_route(self, state: TennisState) -> str:
        """Get query type route from LLM decision."""
        decision = state.get("query_type_decision", {})
        return decision.get("route_to", "data_specific")  # Fallback to data_specific
    
    def _get_tool_route(self, state: TennisState) -> str:
        """Get tool route from LLM decision."""
        decision = state.get("tool_routing_decision", {})
        return decision.get("route_to", "search")  # Fallback to search
    
    def _get_next_step_route(self, state: TennisState) -> str:
        """Get next step route from LLM decision."""
        decision = state.get("next_step_decision", {})
        return decision.get("next_step", "synthesizer")  # Fallback to synthesizer
    
    def process_query(self, user_query: str, session_id: str, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Process a tennis query through the LangGraph workflow."""
        start_time = time.time()
        
        # Start status indicator
        self._start_status_indicator("thinking...")
        
        try:
            self._debug_print(f"\n🚀 LangGraph Processing: '{user_query}'")
            if callbacks:
                self._debug_print(f"📊 Trace capture enabled with {len(callbacks)} callback(s)")
            
            # Load conversation history and build initial state with memory context
            conversation_history = self._build_conversation_context(session_id)
            
            # Initialize state with conversation history + current query for LangGraph memory
            initial_messages = conversation_history + [HumanMessage(content=user_query)]
            initial_state = TennisState(
                messages=initial_messages,
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
            
            # Prepare config with callbacks if provided
            config = {}
            if callbacks:
                config["callbacks"] = callbacks
                self._debug_print("📋 Added callbacks to workflow config")
            
            # Run the workflow with or without callbacks
            if config:
                final_state = self.workflow.invoke(initial_state, config)
            else:
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
            
            # Save conversation to memory (only if successful)
            if not error and response:
                self._save_conversation(session_id, user_query, response)
            
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