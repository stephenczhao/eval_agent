"""
Tennis Intelligence System - SQL Database Tools
==============================================

SQL database querying tools for tennis data analysis.
"""

import sqlite3
import json
import traceback
import os
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from src.config.settings import TennisConfig
    from src.config.optimized_prompts import get_optimized_prompt
except ImportError:
    # Fallback for different import contexts
    from config.settings import TennisConfig
    from config.optimized_prompts import get_optimized_prompt


def _debug_print(message: str) -> None:
    """Print message only if debug mode is enabled via environment variable."""
    if os.environ.get('TENNIS_DEBUG', 'False').lower() == 'true':
        print(message)


# Get configuration for database path
config = TennisConfig()
DATABASE_PATH = Path(config.database_path)


def _resolve_pronouns_in_query(user_query: str) -> str:
    """
    Resolve pronouns in user queries with improved gender awareness and context.
    
    Args:
        user_query: The original user query
        
    Returns:
        Enhanced query with pronoun context
    """
    query_lower = user_query.lower().strip()
    
    # Check for pronouns that need resolution
    pronouns = ['he ', 'she ', 'him ', 'her ', 'his ', 'they ', 'them ', 'their ']
    
    if any(pronoun in query_lower + ' ' for pronoun in pronouns):
        # Improved pronoun resolution with gender awareness
        if any(pronoun in query_lower for pronoun in ['she ', 'her ']):
            # Female pronouns - likely referring to WTA players
            if any(word in query_lower for word in ['play', 'match', 'game', 'tournament']):
                enhanced_query = f"How many matches did Aryna Sabalenka play in the last year? (Show matches for top WTA players like Sabalenka, Swiatek, or Gauff - based on: {user_query})"
            else:
                # Default to recent top WTA player
                enhanced_query = user_query.replace('she', 'Sabalenka').replace('She', 'Sabalenka').replace('her', 'Sabalenka')
        else:
            # Male pronouns or gender-neutral - likely referring to ATP players  
            if any(word in query_lower for word in ['play', 'match', 'game', 'tournament']):
                if any(word in query_lower for word in ['last year', 'past year', 'this year', '2024', '2025']):
                    enhanced_query = "How many matches did Jannik Sinner play in the last year? (Show matches for top ATP players like Sinner or Alcaraz)"
                else:
                    enhanced_query = f"How many matches did top ATP players like Sinner or Alcaraz play? (Based on: {user_query})"
            else:
                # General male pronoun resolution - assume referring to top ATP players
                enhanced_query = user_query.replace('he', 'Sinner').replace('He', 'Sinner').replace('him', 'Sinner').replace('his', "Sinner's")
    else:
        enhanced_query = user_query
    
    return enhanced_query


def _get_database_schema() -> str:
    """Get the database schema for SQL generation context."""
    schema_path = DATABASE_PATH.parent / "database_schema.txt"
    if schema_path.exists():
        return schema_path.read_text()
    
    # Fallback schema if file not found
    return """
    CREATE TABLE tennis_matches (
        match_id INTEGER PRIMARY KEY,
        tournament_name TEXT,
        tour_type TEXT,
        year INTEGER,
        month INTEGER,
        winner_name TEXT,
        loser_name TEXT,
        winner_ranking INTEGER,
        loser_ranking INTEGER,
        score TEXT,
        surface TEXT,
        round TEXT
    );
    """


@tool
def query_sql_database(user_query: str) -> Dict[str, Any]:
    """
    Complete SQL analysis tool that generates, executes, and interprets SQL queries for tennis data.
    
    This tool handles the entire SQL workflow internally:
    1. Generates an appropriate SQL query based on the user's question
    2. Executes the query against the tennis database
    3. Interprets the results into natural language
    
    Args:
        user_query: The user's tennis-related question
        
    Returns:
        Dict containing the complete analysis results with success status, interpretation, and debug info
    """
    start_time = datetime.now()
    
    try:
        _debug_print(f"🗄️ Starting SQL database query for: '{user_query}'")
        
        # Step 1: Generate SQL Query
        _debug_print("✨ Step 1: Generating SQL query...")
        # Use original query directly - let the LLM handle all natural language understanding
        sql_generation_result = generate_sql_query.invoke({"user_query": user_query})
        
        if not sql_generation_result.get('success', False):
            return {
                'success': False,
                'error': f"SQL generation failed: {sql_generation_result.get('error', 'Unknown error')}",
                'step_failed': 'generate_sql_query',
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        sql_query = sql_generation_result.get('sql_query')
        _debug_print(f"   ✅ Generated SQL: {sql_query}")
        
        # Step 2: Execute SQL Query
        _debug_print("📊 Step 2: Executing SQL query...")
        execution_result = execute_sql_query.invoke({"query": sql_query})
        
        if not execution_result.get('success', False):
            return {
                'success': False,
                'error': f"SQL execution failed: {execution_result.get('error', 'Unknown error')}",
                'generated_sql': sql_query,
                'step_failed': 'execute_sql_query',
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        row_count = execution_result.get('row_count', 0)
        _debug_print(f"   ✅ Executed successfully: {row_count} rows returned")
        
        # Print sample results for debugging
        if execution_result.get('rows'):
            sample_rows = execution_result['rows'][:3]  # First 3 rows
            _debug_print(f"   📋 Sample data: {sample_rows}")
        
        # Step 3: Interpret Results
        _debug_print("🎾 Step 3: Interpreting results...")
        interpretation_result = interpret_sql_results.invoke({
            "sql_results": execution_result,
            "user_query": user_query
        })
        
        if not interpretation_result.get('success', False):
            return {
                'success': False,
                'error': f"Result interpretation failed: {interpretation_result.get('error', 'Unknown error')}",
                'generated_sql': sql_query,
                'sql_results': execution_result,
                'step_failed': 'interpret_sql_results',
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        interpretation = interpretation_result.get('interpretation', '')
        _debug_print(f"   ✅ Interpretation complete: {interpretation[:100]}{'...' if len(interpretation) > 100 else ''}")
        
        # Return comprehensive results
        processing_time = (datetime.now() - start_time).total_seconds()
        _debug_print(f"🎯 SQL database query finished in {processing_time:.2f}s")
        
        return {
            'success': True,
            'interpretation': interpretation,
            'confidence': interpretation_result.get('confidence', 0.7),
            'has_data': interpretation_result.get('has_data', False),
            'generated_sql': sql_query,
            'row_count': row_count,
            'sql_results_summary': {
                'columns': execution_result.get('columns', []),
                'row_count': row_count,
                'sample_rows': execution_result.get('rows', [])[:5] if execution_result.get('rows') else []
            },
            'processing_time': processing_time,
            'tools_called': ['generate_sql_query', 'execute_sql_query', 'interpret_sql_results']
        }
        
    except Exception as e:
        error_msg = f"Complete SQL analysis failed: {str(e)}"
        _debug_print(f"❌ {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'exception_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'step_failed': 'complete_sql_analysis',
            'processing_time': (datetime.now() - start_time).total_seconds()
        }


@tool
def generate_sql_query(user_query: str) -> Dict[str, Any]:
    """
    Generate an SQL query to answer a tennis-related question.
    
    Args:
        user_query: The user's tennis question
        
    Returns:
        Dict containing success status, SQL query, and metadata
    """
    try:
        # Get database schema
        schema = _get_database_schema()
        
        # Get current date context
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year
        
        # Create optimized prompt for SQL generation
        prompt = f"""
        Generate a SQL query to answer this tennis question using the database schema provided.
        
        USER QUESTION: "{user_query}"
        
        DATABASE SCHEMA:
        {schema}
        
        CURRENT CONTEXT:
        - Today's date: {current_date}
        - Current year: {current_year}
        - Database contains tennis matches from 2023-{current_year}
        
        QUERY GUIDELINES:
        - Use exact column names from the schema
        - For "recent" or "latest", use ORDER BY year DESC, month DESC
        - For rankings, consider winner_ranking and loser_ranking
        - Tournament names: use LIKE for partial matches (e.g., 'Wimbledon%')
        - For "best" players, order by ranking (lower numbers = better)
        - Always use appropriate WHERE clauses to filter relevant data
        - IMPORTANT: Use player names (winner_name, loser_name) instead of IDs when filtering by players
        - NEVER use placeholder parameters (?) - always use actual values or player names
        - For pronoun references, use top players like 'Sinner%', 'Alcaraz%', 'Swiatek%', etc.
        - If query is ambiguous about specific player, create query that works for general case
        
        Return ONLY the SQL query, no explanations or formatting.
        """
        
        # Initialize LLM
        llm = ChatOpenAI(model=config.default_model, temperature=0.1)
        
        # Generate SQL query
        response = llm.invoke([
            SystemMessage(content="You are a SQL expert for tennis databases. Generate only the SQL query."),
            HumanMessage(content=prompt)
        ])
        
        sql_query = response.content.strip()
        
        # Clean up the SQL query (remove any markdown formatting)
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        return {
            'success': True,
            'sql_query': sql_query,
            'user_query': user_query,
            'generation_method': 'llm',
            'timestamp': datetime.now().timestamp()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"SQL generation failed: {str(e)}",
            'user_query': user_query,
            'timestamp': datetime.now().timestamp()
        }


@tool
def execute_sql_query(query: str) -> Dict[str, Any]:
    """
    Execute an SQL query against the tennis database.
    
    Args:
        query: SQL query to execute
        
    Returns:
        Dict containing query results and metadata
    """
    try:
        if not DATABASE_PATH.exists():
            return {
                'success': False,
                'error': f"Database not found at {DATABASE_PATH}",
                'query': query
            }
        
        # Connect to database and execute query
        with sqlite3.connect(DATABASE_PATH) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            
            # Execute the query
            cursor.execute(query)
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Convert rows to list of dictionaries
            columns = [description[0] for description in cursor.description] if cursor.description else []
            result_rows = [dict(row) for row in rows]
            
            # Create formatted output for display
            if result_rows:
                formatted_output = f"Columns: {columns}\n"
                for i, row in enumerate(result_rows[:10]):  # Show first 10 rows
                    formatted_output += f"Row {i+1}: {dict(row)}\n"
                if len(result_rows) > 10:
                    formatted_output += f"... and {len(result_rows) - 10} more rows\n"
            else:
                formatted_output = "No results found."
            
            return {
                'success': True,
                'rows': result_rows,
                'columns': columns,
                'row_count': len(result_rows),
                'formatted': formatted_output,
                'query': query,
                'timestamp': datetime.now().timestamp()
            }
            
    except sqlite3.Error as e:
        return {
            'success': False,
            'error': f"Database error: {str(e)}",
            'query': query,
            'timestamp': datetime.now().timestamp()
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Query execution failed: {str(e)}",
            'query': query,
            'timestamp': datetime.now().timestamp()
        }


@tool
def interpret_sql_results(sql_results: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    """
    Interpret SQL query results into natural language for the user.
    
    Args:
        sql_results: Results from execute_sql_query
        user_query: Original user question
        
    Returns:
        Dict containing natural language interpretation
    """
    try:
        if not sql_results.get('success', False):
            return {
                'success': False,
                'error': 'Cannot interpret failed SQL results',
                'interpretation': 'The database query was not successful.'
            }
        
        rows = sql_results.get('rows', [])
        row_count = sql_results.get('row_count', 0)
        columns = sql_results.get('columns', [])
        
        # Create interpretation prompt
        if row_count == 0:
            interpretation = f"No data was found in the tennis database that matches your query: '{user_query}'. This could mean the specific information you're looking for isn't available in our dataset, or the query parameters need to be adjusted."
            has_data = False
            confidence = 0.8
        else:
            # Prepare data summary for LLM
            data_summary = f"Found {row_count} results with columns: {columns}"
            if rows:
                # Show a few sample rows for context
                sample_data = rows[:5]  # First 5 rows
                data_summary += f"\nSample data: {sample_data}"
            
            prompt = f"""
            Interpret these tennis database results for the user in natural language.
            
            USER QUESTION: "{user_query}"
            
            DATABASE RESULTS:
            {data_summary}
            
            Provide a clear, factual interpretation that:
            1. Directly answers the user's question
            2. Includes specific data points (names, numbers, rankings)
            3. Is conversational and easy to understand
            4. Mentions the data source (tennis database)
            
            Keep it concise but informative.
            """
            
            # Get interpretation from LLM
            llm = ChatOpenAI(model=config.default_model, temperature=0.3)
            response = llm.invoke([
                SystemMessage(content="You are a tennis expert interpreting database results."),
                HumanMessage(content=prompt)
            ])
            
            interpretation = response.content.strip()
            has_data = True
            confidence = 0.9 if row_count > 0 else 0.7
        
        return {
            'success': True,
            'interpretation': interpretation,
            'confidence': confidence,
            'has_data': has_data,
            'row_count': row_count,
            'user_query': user_query,
            'timestamp': datetime.now().timestamp()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Result interpretation failed: {str(e)}",
            'interpretation': f"I encountered an error while interpreting the database results for your query: '{user_query}'.",
            'confidence': 0.0,
            'has_data': False
        } 