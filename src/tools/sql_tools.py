"""
Tennis Intelligence System - SQL Database Tools
==============================================

SQL database querying tools for tennis data analysis.
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


from src.config.settings import TennisConfig
from src.config.optimized_prompts import get_optimized_prompt


# Get configuration for database path
config = TennisConfig()
DATABASE_PATH = Path(config.database_path)


def _get_database_schema() -> str:
    """Get the database schema for SQL generation context - MUST use actual schema file."""
    schema_path = DATABASE_PATH.parent / "database_schema.txt"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Database schema file not found: {schema_path}. Cannot proceed without actual schema.")
    
    try:
        schema_content = schema_path.read_text(encoding='utf-8')
        if not schema_content.strip():
            raise ValueError(f"Database schema file is empty: {schema_path}")
        return schema_content
    except Exception as e:
        raise Exception(f"Failed to load database schema from {schema_path}: {str(e)}")


def _execute_sql_query(sql_query: str) -> Dict[str, Any]:
    """
    Execute SQL query against the tennis database.
    
    Args:
        sql_query: SQL query to execute
        
    Returns:
        Dict containing query results and metadata
    """
    if not DATABASE_PATH.exists():
        return {
            "success": False,
            "error": f"Database file not found: {DATABASE_PATH}",
            "results": [],
            "query": sql_query
        }
    
    try:
        # Connect to database with timeout
        conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
        cursor = conn.cursor()
        
        # Execute query
        start_time = time.time()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        execution_time = time.time() - start_time
        
        # Get column names
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        # Convert results to list of dictionaries
        formatted_results = []
        for row in results:
            formatted_results.append(dict(zip(column_names, row)))
        
        conn.close()
        
        return {
            "success": True,
            "results": formatted_results,
            "column_names": column_names,
            "row_count": len(results),
            "execution_time": execution_time,
            "query": sql_query
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "error": f"SQL Error: {str(e)}",
            "results": [],
            "query": sql_query
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "results": [],
            "query": sql_query
                } 


def execute_sql_query(sql_query: str) -> Dict[str, Any]:
    """
    Execute SQL query against the tennis database.
    
    Args:
        sql_query: SQL query to execute
        
    Returns:
        Dict containing query results and metadata
    """
    return _execute_sql_query(sql_query)


def generate_sql_query(user_query: str) -> Dict[str, Any]:
    """
    Generate an SQL query to answer a tennis-related question.
    
    Args:
        user_query: The user's tennis question
        
    Returns:
        Dict containing success status, SQL query, and metadata
    """
    try:
        # Get database schema - will raise exception if schema file not found
        try:
            schema = _get_database_schema()
        except Exception as schema_error:
            return {
                "success": False,
                "error": f"Schema loading failed: {str(schema_error)}",
                "sql_query": "",
                "original_query": user_query,
                "query_type": "schema_error",
                "confidence": 0.0
            }
        
        # Get current date context
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year
        
        # Create context-rich prompt for SQL generation
        prompt = f"""
        Generate a SQL query to answer this tennis question using the database schema provided.
        
        USER QUESTION: "{user_query}"
        
        DATABASE SCHEMA:
        {schema}
        
        CRITICAL SQL GENERATION RULES:
        
        1. PLAYER NAMES: Use EXACT format from database - players are stored as "Surname FirstInitial." (e.g., "Sabalenka A.", "Djokovic N.")
        
        2. WIN RATIO CALCULATIONS: Calculate from tennis_matches table:
           ```sql
           SELECT 
             SUM(CASE WHEN winner_name = 'Player Name' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_ratio
           FROM tennis_matches 
           WHERE (winner_name = 'Player Name' OR loser_name = 'Player Name') AND year = YYYY;
           ```
        
        3. AVAILABLE TABLES:
           - tennis_matches: Individual match records
           - players: Aggregated player stats  
           - PLAYER_MATCH_STATS: View with win percentages
           - HEAD_TO_HEAD: View for head-to-head records
           - SURFACE_PERFORMANCE: View for surface-specific stats
        
        4. COMMON NAME MAPPINGS:
           - "Sabalenka" → "Sabalenka A."
           - "Djokovic" → "Djokovic N." 
           - "Nadal" → "Nadal R."
           - "Federer" → "Federer R."
        
        5. YEAR FILTERING: Use year column for temporal queries
        
        6. RANKINGS: Lower numbers = better rankings (1 = #1 in world)
        
        CONTEXT:
        - Today: {current_date}
        - Database coverage: 2023-01-01 to 2025-06-28
        - 13,303 matches total (ATP: 6,899, WTA: 6,404)
        
        Return ONLY the SQL query, no explanations or markdown formatting.
        """
        
        # Initialize LLM
        llm = ChatOpenAI(model=config.default_model, temperature=0.1)
        
        # Generate SQL query
        response = llm.invoke([
            SystemMessage(content="You are a SQL query generator for a tennis database. Generate accurate SQLite queries."),
            HumanMessage(content=prompt)
        ])
        
        sql_query = str(response.content).strip()
        
        # Clean up the SQL query (remove markdown formatting if present)
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        return {
            "success": True,
            "sql_query": sql_query,
            "original_query": user_query,
            "query_type": "generated",
            "confidence": 0.8
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"SQL generation failed: {str(e)}",
            "sql_query": "",
            "original_query": user_query,
            "query_type": "failed",
            "confidence": 0.0
        }


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
        
        rows = sql_results.get('results', []) # Changed from 'rows' to 'results'
        row_count = sql_results.get('row_count', 0)
        columns = sql_results.get('column_names', []) # Changed from 'columns' to 'column_names'
        
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
            
            interpretation = str(response.content).strip()
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


@tool
def query_sql_database(user_query: str) -> Dict[str, Any]:
    """
    Complete SQL database querying tool that generates, executes, and interprets SQL queries.
    
    Args:
        user_query: The user's tennis question
        
    Returns:
        Dict containing complete SQL analysis results
    """
    try:
        # Step 1: Generate SQL query
        sql_gen_result = generate_sql_query(user_query)
        if not sql_gen_result.get('success', False):
            error_type = sql_gen_result.get('query_type', 'unknown')
            if error_type == 'schema_error':
                return {
                    'success': False,
                    'error': f"Database schema error: {sql_gen_result.get('error', 'Schema not accessible')}",
                    'user_query': user_query,
                    'step_failed': 'schema_loading',
                    'schema_error': True
                }
            else:
                return {
                    'success': False,
                    'error': f"SQL generation failed: {sql_gen_result.get('error', 'Unknown error')}",
                    'user_query': user_query,
                    'step_failed': 'generation'
                }
        
        sql_query = sql_gen_result['sql_query']
        
        # Step 2: Execute SQL query
        sql_exec_result = execute_sql_query(sql_query)
        if not sql_exec_result.get('success', False):
            return {
                'success': False,
                'error': f"SQL execution failed: {sql_exec_result.get('error', 'Unknown error')}",
                'user_query': user_query,
                'sql_query': sql_query,
                'step_failed': 'execution'
            }
        
        # Step 3: Interpret results
        interp_result = interpret_sql_results(sql_exec_result, user_query)
        if not interp_result.get('success', False):
            return {
                'success': False,
                'error': f"Result interpretation failed: {interp_result.get('error', 'Unknown error')}",
                'user_query': user_query,
                'sql_query': sql_query,
                'sql_results': sql_exec_result,
                'step_failed': 'interpretation'
            }
        
        # Combine all results
        return {
            'success': True,
            'user_query': user_query,
            'sql_query': sql_query,
            'raw_results': sql_exec_result['results'],
            'row_count': sql_exec_result['row_count'],
            'interpretation': interp_result['interpretation'],
            'confidence': interp_result['confidence'],
            'has_data': interp_result['has_data'],
            'execution_time': sql_exec_result.get('execution_time', 0),
            'timestamp': datetime.now().timestamp()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Unexpected error in SQL processing: {str(e)}",
            'user_query': user_query,
            'step_failed': 'unexpected'
        } 