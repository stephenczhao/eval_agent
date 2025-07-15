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
    refined_query: str = Field(description="Refined or clarified version of the query")
    reasoning: str = Field(description="Brief explanation of the classification decision")


class RoutingResult(BaseModel):
    """Tennis query routing decision."""
    sql_needed: bool = Field(description="Whether SQL database query is needed")
    search_needed: bool = Field(description="Whether web search is needed")
    reasoning: str = Field(description="Explanation of routing decision")
    priority: Literal["sql_first", "search_first", "parallel"] = Field(
        description="Which data source to prioritize"
    )


class TennisState(TypedDict):
    """State for the tennis intelligence workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    refined_query: str
    routing_decision: Dict[str, Any]
    sql_results: Optional[Dict[str, Any]]
    search_results: Optional[Dict[str, Any]]
    final_response: Optional[str]
    confidence: float
    sources: List[str]
    error: Optional[str]
    session_id: str
    # Simple session memory
    mentioned_players: List[str]
    conversation_context: Optional[str]


class LangGraphTennisOrchestrator:
    """
    LangGraph-based Tennis Intelligence Orchestrator using official tool calling patterns.
    
    Uses StateGraph to manage workflow with proper tool calling through ToolNode.
    """
    
    def __init__(self, config: TennisConfig, debug: bool = False):
        """Initialize the LangGraph orchestrator."""
        self.config = config
        self.debug = debug
        
        # Simple session memory storage
        self._session_memory = {}
        
        # Initialize LLM with tool binding
        self.llm = ChatOpenAI(
            model=config.default_model,
            temperature=config.temperature,
            api_key=config.openai_api_key
        )
        
        # Define available tools - only 2 main tools now
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
        
        # Loading animation setup
        self._loading_active = False
        self._loading_thread = None

    def _debug_print(self, message: str) -> None:
        """Print message only if debug mode is enabled."""
        if self.debug:
            print(message)

    def _start_loading_animation(self):
        """Start the thinking... loading animation"""
        if not self.debug and not self._loading_active:
            self._loading_active = True
            self._loading_thread = threading.Thread(target=self._run_loading_animation)
            self._loading_thread.daemon = True
            self._loading_thread.start()

    def _stop_loading_animation(self):
        """Stop the loading animation"""
        if self._loading_active:
            self._loading_active = False
            if self._loading_thread:
                self._loading_thread.join(timeout=0.1)
            # Clear the line
            if not self.debug:
                print("\r" + " " * 20 + "\r", end="", flush=True)

    def _run_loading_animation(self):
        """Run the loading animation in a separate thread"""
        spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        while self._loading_active:
            print(f"\r{next(spinner)} thinking...", end="", flush=True)
            time.sleep(0.1)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for tennis query processing."""
        
        # Create the state graph
        workflow = StateGraph(TennisState)
        
        # Add nodes
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("router", self._route_query)
        workflow.add_node("sql_agent", self._sql_agent)
        workflow.add_node("search_agent", self._search_agent)
        workflow.add_node("synthesizer", self._synthesize_response)
        workflow.add_node("tools", self.tool_node)
        
        # Define the workflow flow
        workflow.set_entry_point("classifier")
        
        # Classifier routes to router
        workflow.add_edge("classifier", "router")
        
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
        
        # SQL agent can call tools or go to search (if both needed) or synthesizer
        workflow.add_conditional_edges(
            "sql_agent",
            self._sql_next_step,
            {
                "tools": "tools",
                "search_agent": "search_agent",
                "synthesizer": "synthesizer"
            }
        )
        
        # Search agent can call tools or go to synthesizer
        workflow.add_conditional_edges(
            "search_agent", 
            self._search_next_step,
            {
                "tools": "tools",
                "synthesizer": "synthesizer"
            }
        )
        
        # Tools route back to appropriate agent based on last tool called
        workflow.add_conditional_edges(
            "tools",
            self._tools_next_step,
            {
                "sql_agent": "sql_agent",
                "search_agent": "search_agent",
                "synthesizer": "synthesizer"
            }
        )
        
        # Synthesizer ends the workflow
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    def _get_session_memory(self, session_id: str) -> Dict[str, Any]:
        """Get session memory for the given session ID."""
        if session_id not in self._session_memory:
            self._session_memory[session_id] = {
                "mentioned_players": [],
                "last_player_mentioned": None,
                "conversation_history": []
            }
        return self._session_memory[session_id]
    
    def _update_session_memory(self, session_id: str, user_query: str, response: str, players_mentioned: List[str]) -> None:
        """Update session memory with new conversation data."""
        memory = self._get_session_memory(session_id)
        
        # Update conversation history (keep last 5 exchanges)
        memory["conversation_history"].append({
            "query": user_query,
            "response": response[:200],  # Truncate for memory efficiency
            "timestamp": datetime.now().isoformat()
        })
        if len(memory["conversation_history"]) > 5:
            memory["conversation_history"] = memory["conversation_history"][-5:]
        
        # Update mentioned players
        for player in players_mentioned:
            if player not in memory["mentioned_players"]:
                memory["mentioned_players"].append(player)
        
        # Keep track of most recently mentioned player
        if players_mentioned:
            memory["last_player_mentioned"] = players_mentioned[-1]
    
    def _extract_players_from_response(self, response: str) -> List[str]:
        """Extract player names from response text."""
        players = []
        
        # Common tennis players to look for
        known_players = [
            "Sinner", "Jannik Sinner", "Alcaraz", "Carlos Alcaraz", 
            "Djokovic", "Novak Djokovic", "Nadal", "Rafael Nadal",
            "Federer", "Roger Federer", "Sabalenka", "Aryna Sabalenka",
            "Swiatek", "Iga Swiatek", "Gauff", "Coco Gauff",
            "Zverev", "Alexander Zverev", "Medvedev", "Daniil Medvedev"
        ]
        
        response_lower = response.lower()
        for player in known_players:
            if player.lower() in response_lower:
                # Use the short form for consistency
                short_name = player.split()[-1]  # Get last name
                if short_name not in players:
                    players.append(short_name)
        
        return players
    
    def _classify_query(self, state: TennisState) -> TennisState:
        """Classify and refine the tennis query using structured output."""
        self._debug_print(f"🎾 Classifying query: '{state['user_query']}'")
        
        # Get session memory for context
        session_memory = self._get_session_memory(state.get("session_id", "default"))
        mentioned_players = session_memory.get("mentioned_players", [])
        last_player = session_memory.get("last_player_mentioned")
        conversation_history = session_memory.get("conversation_history", [])
        
        # Update state with memory context
        state["mentioned_players"] = mentioned_players
        state["conversation_context"] = f"Recent players mentioned: {mentioned_players}" if mentioned_players else "No recent context"
        
        # Create LLM with structured output
        classifier_llm = self.llm.with_structured_output(ClassificationResult)
        
        # Build context for pronoun resolution
        context_info = ""
        if mentioned_players:
            context_info += f"\nRECENT PLAYERS MENTIONED: {', '.join(mentioned_players)}"
        if last_player:
            context_info += f"\nLAST PLAYER MENTIONED: {last_player}"
        if conversation_history:
            recent_context = conversation_history[-1] if conversation_history else None
            if recent_context:
                context_info += f"\nPREVIOUS QUERY: {recent_context['query']}"
        
        # Create classification prompt with memory context
        classifier_prompt = f"""
        You are a tennis query classifier for a tennis intelligence system. Analyze this query:
        
        QUERY: "{state['user_query']}"
        {context_info}
        
        CONTEXT: This is a tennis-focused system, so assume ambiguous queries about "players", "rankings", 
        "matches", "tournaments" are tennis-related unless clearly specified otherwise.
        
        PRONOUN RESOLUTION: 
        - If the query uses pronouns like "he", "him", "his" and there are recently mentioned MALE players, 
          substitute with the most relevant male player name.
        - If the query uses pronouns like "she", "her" and there are recently mentioned FEMALE players,
          substitute with the most relevant female player name.
        - GENDER MAPPING: Sinner, Alcaraz, Djokovic, Nadal, Federer, Zverev, Medvedev = MALE
        - GENDER MAPPING: Sabalenka, Swiatek, Gauff = FEMALE
        - For "they/them", use context to determine if referring to multiple players or gender-neutral
        
        Examples:
        - "How many games did he play?" + Recent: [Sinner, Sabalenka] → "How many games did Sinner play?"
        - "Who did she beat?" + Recent: [Alcaraz, Swiatek] → "Who did Swiatek beat?"
        - "What's his ranking?" + Recent: [Sinner, Alcaraz] → "What's Sinner's ranking?" (most recent male)
        
        Examples of tennis queries:
        - "Who's the best player right now?" → Tennis-related
        - "Who won the latest tournament?" → Tennis-related  
        - "Current rankings" → Tennis-related
        - "Player stats" → Tennis-related
        - "What's the weather?" → Not tennis-related
        
        Determine if this is tennis-related and refine/clarify the query if needed.
        Be generous in classifying queries as tennis-related when the context is ambiguous.
        If pronouns are used and recent players are known, substitute them appropriately based on gender.
        """
        
        messages = [
            SystemMessage(content="You are a tennis query classifier for a tennis intelligence system. Be generous in classifying ambiguous queries as tennis-related since this is a tennis-focused system."),
            HumanMessage(content=classifier_prompt)
        ]
        
        try:
            result = classifier_llm.invoke(messages)
            
            if result.is_tennis_related:
                state["refined_query"] = result.refined_query
                self._debug_print(f"✅ Tennis query confirmed: {state['refined_query']}")
            else:
                state["error"] = "Non-tennis query detected"
                state["final_response"] = "I'm designed for tennis-related questions only."
                self._debug_print(f"❌ Non-tennis query: {result.reasoning}")
                
        except Exception as e:
            state["refined_query"] = state["user_query"]  # Fallback
            self._debug_print(f"⚠️ Classification failed ({str(e)}), proceeding with original query")
        
        return state
    
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
        current_year = datetime.now().year
        routing_prompt = f"""
        Analyze this tennis query and determine optimal routing:
        
        QUERY: "{state['refined_query']}"
        ENTITIES: {entities}
        CURRENT YEAR: {current_year}
        
        SOURCES:
        - SQL Database: 2023-{current_year} tennis matches, stats, rankings (historical data)
        - Web Search: Current rankings, recent news, live updates, latest standings
        
        ROUTING GUIDELINES:
        - Keywords like "current", "latest", "right now", "best player now", "top ranked" → FAVOR SEARCH
        - Questions about specific historical matches, past tournaments → FAVOR SQL
        - Player career stats, head-to-head records → FAVOR SQL  
        - Current form, recent performance, live rankings → FAVOR SEARCH
        - "Who won X tournament?" (past events) → SQL
        - "Who's the best/top player?" (current) → SEARCH
        
        Determine which data sources are needed and their priority.
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
            # Intelligent default routing based on query keywords
            query_lower = state['refined_query'].lower()
            current_keywords = ['current', 'latest', 'right now', 'now', 'best player', 'top player', 'ranking', 'ranked']
            
            if any(keyword in query_lower for keyword in current_keywords):
                state["routing_decision"] = {
                    "sql_needed": False,
                    "search_needed": True,
                    "reasoning": f"Default search routing for current info query (structured output failed: {str(e)})",
                    "priority": "search_first"
                }
                self._debug_print("🔍 Defaulting to search for current info query")
            else:
                state["routing_decision"] = {
                    "sql_needed": True,
                    "search_needed": False,
                    "reasoning": f"Default SQL routing for historical query (structured output failed: {str(e)})",
                    "priority": "sql_first"
                }
                self._debug_print("🗄️ Defaulting to SQL for historical query")
        
        return state
    
    def _sql_agent(self, state: TennisState) -> TennisState:
        """SQL agent that uses tool calling for database operations."""
        if state.get("error"):
            return state
            
        self._debug_print("🗄️ SQL Agent executing...")
        
        # Simple and direct: just call the SQL database query tool
        sql_prompt = f"""
        Answer this tennis query using the SQL database query tool.
        
        QUERY: "{state['refined_query']}"
        
        Call query_sql_database with the user query to get a complete answer from the tennis database.
        """
        
        messages = [
            SystemMessage(content="You are a tennis SQL agent. Use the query_sql_database tool to answer tennis questions."),
            HumanMessage(content=sql_prompt)
        ]
        
        # This will trigger tool calling
        response = self.llm_with_tools.invoke(messages)
        state["messages"].append(response)
        
        # Debug: Print tool calls that were made
        if hasattr(response, 'tool_calls') and response.tool_calls:
            self._debug_print(f"🔧 SQL Agent called {len(response.tool_calls)} tools:")
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                self._debug_print(f"   📞 Calling: {tool_name}")
        
        return state
    
    def _search_agent(self, state: TennisState) -> TennisState:
        """Search agent that uses tool calling for web search."""
        if state.get("error"):
            return state
            
        self._debug_print("🌐 Search Agent executing...")
        
        # Prepare message for LLM with tool calling
        search_prompt = f"""
        You need to answer this tennis query using online search:
        
        QUERY: "{state['refined_query']}"
        
        Call the online_search tool to get current tennis information from the web.
        """
        
        messages = [
            SystemMessage(content="You are a tennis search agent. Use the online_search tool to get current tennis information."),
            HumanMessage(content=search_prompt)
        ]
        
        # This will trigger tool calling if the LLM decides to call tools
        response = self.llm_with_tools.invoke(messages)
        state["messages"].append(response)
        
        # Debug: Print tool calls that were made
        if hasattr(response, 'tool_calls') and response.tool_calls:
            self._debug_print(f"🔧 Search Agent called {len(response.tool_calls)} tools:")
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                self._debug_print(f"   📞 Calling: {tool_name}")
        
        return state
    
    def _synthesize_response(self, state: TennisState) -> TennisState:
        """Synthesize final response from collected data."""
        if state.get("error"):
            return state
            
        self._debug_print("🧮 Synthesizing final response...")
        
        # Debug: Print tool results for debugging
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
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        Create a comprehensive tennis response for this query:
        
        QUERY: "{state['refined_query']}"
        
        AVAILABLE INFORMATION:
        {chr(10).join(all_info[-5:])}  # Last 5 pieces of info
        
        SOURCES USED: {list(set(sources))}
        
        Provide a clear, factual response that directly answers the query.
        Keep it concise but informative.
        """
        
        messages = [
            SystemMessage(content=f"You are a tennis expert. Today is {datetime.now().strftime('%Y-%m-%d')}."),
            HumanMessage(content=synthesis_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        state["final_response"] = response.content
        state["sources"] = list(set(sources)) if sources else ["System"]
        state["confidence"] = 0.8 if sources else 0.5
        
        # Extract players mentioned in the final response and update session memory
        players_mentioned = self._extract_players_from_response(response.content)
        session_id = state.get("session_id", "default")
        
        # Update session memory with this conversation
        self._update_session_memory(
            session_id=session_id,
            user_query=state["user_query"],
            response=response.content,
            players_mentioned=players_mentioned
        )
        
        if players_mentioned:
            self._debug_print(f"🧠 Updated session memory with players: {players_mentioned}")
        
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
        last_message = state["messages"][-1] if state["messages"] else None
        
        # If SQL agent decided to call tools, go to tools
        if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Check if we need search too based on routing
        routing = state.get("routing_decision", {})
        if routing.get("search_needed", False):
            return "search_agent"
        
        # Otherwise go straight to synthesis
        return "synthesizer"
    
    def _search_next_step(self, state: TennisState) -> str:
        """Determine next step after search agent."""
        last_message = state["messages"][-1] if state["messages"] else None
        
        if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        return "synthesizer"
    
    def _tools_next_step(self, state: TennisState) -> str:
        """Determine next step after tools execution."""
        # After tools are executed, check which agents have run by looking at messages
        routing = state.get("routing_decision", {})
        sql_needed = routing.get("sql_needed", False)
        search_needed = routing.get("search_needed", False)
        
        # Check which agents have already executed by looking at the messages
        sql_agent_ran = False
        search_agent_ran = False
        
        for message in state["messages"]:
            if hasattr(message, 'content') and message.content:
                content = str(message.content).lower()
                if "sql agent" in content or "database" in content:
                    sql_agent_ran = True
                elif "search agent" in content or "web search" in content:
                    search_agent_ran = True
        
        # If both SQL and search are needed, and only one has run, continue to the other
        if sql_needed and search_needed:
            if sql_agent_ran and not search_agent_ran:
                return "search_agent"
            elif search_agent_ran and not sql_agent_ran:
                return "sql_agent"
        
        # Otherwise, go to synthesis (tools have executed, we're done)
        return "synthesizer"
    
    def process_query(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """Process a tennis query through the LangGraph workflow."""
        start_time = time.time()
        
        # Start loading animation
        self._start_loading_animation()
        
        try:
            self._debug_print(f"\n🚀 LangGraph Processing: '{user_query}'")
            
            # Initialize state
            initial_state = TennisState(
                messages=[],
                user_query=user_query,
                refined_query="",
                routing_decision={},
                sql_results=None,
                search_results=None,
                final_response=None,
                confidence=0.0,
                sources=[],
                error=None,
                session_id=session_id,
                mentioned_players=[],
                conversation_context=None
            )
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Extract results
            response = final_state.get("final_response", "I couldn't process your tennis query.")
            confidence = final_state.get("confidence", 0.0)
            sources = final_state.get("sources", ["System"])
            error = final_state.get("error")
            
            processing_time = time.time() - start_time
            
            return {
                'response': response,
                'confidence': confidence,
                'sources': sources,
                'sql_data_used': 'Tennis Database' in sources,
                'search_data_used': 'Web Search' in sources,
                'error': error is not None,
                'langgraph_used': True,
                'processing_time': processing_time
            }
            
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
            # Stop loading animation
            self._stop_loading_animation() 