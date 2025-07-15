"""
Tennis Intelligence System - SQL Tools
======================================

SQL tools for querying the tennis database and providing structured analysis.
Integrates with the existing run_sql.py utility function.
"""

import time
import re
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import the existing SQL utility
import sys
from pathlib import Path

# Add utils directory to path to import run_sql
utils_path = Path(__file__).parent.parent / "utils"
sys.path.append(str(utils_path))

try:
    from run_sql import run_sql_query
    from src.config.settings import TennisConfig, get_database_schema_context
    from src.config.optimized_prompts import get_optimized_prompt
except ImportError:
    # Fallback import paths
    from utils.run_sql import run_sql_query
    from config.settings import TennisConfig, get_database_schema_context
    from config.optimized_prompts import get_optimized_prompt


@tool
def generate_sql_query(user_query: str) -> Dict[str, Any]:
    """
    Generate SQL query from natural language tennis question.
    
    Args:
        user_query: Natural language question about tennis
        
    Returns:
        Dictionary with generated SQL query and metadata
    """
    config = TennisConfig()
    
    try:
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.1,
            max_tokens=400,
            api_key=config.openai_api_key
        )
        
        # Use optimized SQL prompt with datetime context
        sql_prompt = get_optimized_prompt('sql')
        
        # Create the prompt with the user query
        full_prompt = f"{sql_prompt}\n\nQUESTION: {user_query}\n\nSQL:"

        response = llm.invoke([
            SystemMessage(content=sql_prompt),
            HumanMessage(content=full_prompt)
        ])
        
        sql_query = response.content.strip()
        
        # Enhanced SQL query cleanup - extract only the SQL statement
        if sql_query.startswith('```sql'):
            sql_query = sql_query[6:]
        if sql_query.endswith('```'):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip()
        
        # Look for SQL keywords to extract just the SQL part
        import re
        sql_keywords = r'^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)'
        
        # Split by lines and find the first line starting with SQL keyword
        lines = sql_query.split('\n')
        sql_lines = []
        found_sql = False
        
        for line in lines:
            line = line.strip()
            if not found_sql and re.match(sql_keywords, line, re.IGNORECASE):
                found_sql = True
                sql_lines.append(line)
            elif found_sql:
                # Continue adding lines until we hit a non-SQL line or end
                if line and not line.startswith('--') and not line.startswith('/*'):
                    sql_lines.append(line)
                elif not line:  # Empty line might indicate end of query
                    break
        
        # If we found SQL lines, use them; otherwise use the original
        if sql_lines:
            sql_query = '\n'.join(sql_lines)
        
        # Final cleanup - remove any trailing semicolon issues
        sql_query = sql_query.strip().rstrip(';') + ';'
        
        return {
            "success": True,
            "sql_query": sql_query,
            "user_query": user_query,
            "generation_method": "llm",
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "success": False,
            "sql_query": "",
            "user_query": user_query,
            "error": f"SQL generation failed: {str(e)}",
            "generation_method": "failed",
            "timestamp": time.time()
        }


@tool
def execute_sql_query(query: str, max_rows: int = 100) -> Dict[str, Any]:
    """
    Execute SQL query against the tennis database.
    
    Args:
        query: SQL query to execute
        max_rows: Maximum number of rows to return
        
    Returns:
        Dictionary with query results and metadata
    """
    config = TennisConfig()
    start_time = time.time()
    
    try:
        # Use the existing run_sql_query function
        results = run_sql_query(query, max_rows=max_rows)
        
        execution_time = time.time() - start_time
        
        if results.get("success", False):
            return {
                "success": True,
                "columns": results.get("columns", []),
                "rows": results.get("rows", []),
                "row_count": len(results.get("rows", [])),
                "formatted": results.get("formatted", ""),
                "execution_time": execution_time,
                "query_used": query,
                "timestamp": time.time()
            }
        else:
            return {
                "success": False,
                "error": results.get("error", "Unknown error"),
                "execution_time": execution_time,
                "query_used": query,
                "timestamp": time.time()
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Query execution failed: {str(e)}",
            "execution_time": time.time() - start_time,
            "query_used": query,
            "timestamp": time.time()
        }


@tool
def interpret_sql_results(sql_results: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    """
    Interpret SQL results into natural language using efficient LLM processing.
    
    Args:
        sql_results: Results from execute_sql_query
        user_query: Original user question for context
        
    Returns:
        Dictionary with natural language interpretation
    """
    config = TennisConfig()
    
    if not sql_results.get("success", False):
        return {
            "success": False,
            "interpretation": f"Query failed: {sql_results.get('error', 'Unknown error')}",
            "confidence": 0.0,
            "has_data": False
        }
    
    rows = sql_results.get("rows", [])
    columns = sql_results.get("columns", [])
    
    if not rows:
        return {
            "success": True,
            "interpretation": "No data found matching your query criteria.",
            "confidence": 0.9,
            "has_data": False
        }
    
    try:
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.2,
            max_tokens=300,
            api_key=config.openai_api_key
        )
        
        # Create concise data summary for LLM
        data_summary = f"Columns: {', '.join(columns)}\n"
        data_summary += f"Rows found: {len(rows)}\n"
        
        # Include first few rows of data
        for i, row in enumerate(rows[:3]):
            data_summary += f"Row {i+1}: {dict(zip(columns, row))}\n"
        
        if len(rows) > 3:
            data_summary += f"... and {len(rows) - 3} more rows\n"
        
        # Use optimized prompt for interpretation with datetime context
        sql_prompt = get_optimized_prompt('sql')
        interpretation_prompt = f"""USER QUESTION: "{user_query}"\n\nDATABASE RESULTS:\n{data_summary}\n\nIMPORTANT: winner_points and loser_points are ATP ranking points, NOT match scores. If asked for match scores and only points are available, explain that match scores aren't in the database.\n\nProvide a clear, conversational answer in 2-3 sentences focusing on direct answer, key findings, and specific numbers/statistics:"""

        response = llm.invoke([
            SystemMessage(content=sql_prompt),
            HumanMessage(content=interpretation_prompt)
        ])
        
        interpretation = response.content.strip()
        
        return {
            "success": True,
            "interpretation": interpretation,
            "confidence": 0.9,
            "has_data": True
        }
        
    except Exception as e:
        # Create fallback interpretation
        fallback_interpretation = f"Found {len(rows)} records in the database matching your query about {user_query}."
        
        return {
            "success": True,
            "interpretation": fallback_interpretation,
            "confidence": 0.7,
            "has_data": True
        } 