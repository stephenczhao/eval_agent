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
except ImportError:
    # Fallback import paths
    from utils.run_sql import run_sql_query
    from config.settings import TennisConfig, get_database_schema_context


def generate_sql_query(user_query: str) -> Dict[str, Any]:
    """
    Generate SQL query from natural language using LLM with database schema context.
    
    Args:
        user_query: Natural language tennis question
        
    Returns:
        Dictionary containing generated SQL query and metadata
    """
    config = TennisConfig()
    
    try:
        llm = ChatOpenAI(model=config.default_model, temperature=0.1)
        
        schema_context = get_database_schema_context()
        
        sql_generation_prompt = f"""You are a SQL expert for a tennis intelligence system. Generate a precise SQL query to answer the user's question.

DATABASE SCHEMA:
{schema_context}

USER QUESTION: "{user_query}"

IMPORTANT GUIDELINES:
1. Generate ONLY the SQL query, no explanations
2. Use proper table joins when needed
3. Handle player names with LIKE for partial matches (e.g., WHERE player1 LIKE '%Sinner%' OR player2 LIKE '%Sinner%')
4. For "last year" or "2024", use WHERE match_date conditions appropriately
5. Use COUNT(*) for counting matches between players
6. Include proper date filtering for temporal queries
7. Order results logically (e.g., by date, score, etc.)
8. Limit results when appropriate (e.g., LIMIT 10 for large result sets)

EXAMPLE PATTERNS:
- "How many times did X play Y?" → COUNT matches WHERE both players involved
- "X's win ratio in 2024" → Calculate wins/total matches for that year
- "X vs Y head-to-head" → All matches between the two players
- "Who won the latest tournament?" → Most recent tournament winner

Generate the SQL query:"""

        response = llm.invoke([
            SystemMessage(content="You are a SQL expert. Generate precise SQL queries for tennis database questions. Return ONLY the SQL query."),
            HumanMessage(content=sql_generation_prompt)
        ])
        
        sql_query = response.content.strip()
        
        # Clean up the response (remove any markdown formatting)
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        
        return {
            "success": True,
            "sql_query": sql_query,
            "user_query": user_query,
            "method": "llm_generated"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"SQL generation failed: {str(e)}",
            "sql_query": None,
            "user_query": user_query,
            "method": "failed"
        }


@tool
def execute_sql_query(query: str, max_rows: int = 100) -> Dict[str, Any]:
    """
    Execute a SQL query against the tennis matches database.
    
    This tool executes SQL queries against the comprehensive tennis database
    containing match results, player statistics, and tournament information
    from 2023-2025.
    
    Args:
        query: SQL query to execute
        max_rows: Maximum number of rows to return (default 100)
        
    Returns:
        Dictionary containing query results, success status, and metadata
    """
    config = TennisConfig()
    start_time = time.time()
    
    try:
        # Execute query using the existing utility
        result = run_sql_query(
            query=query,
            db_path=config.database_path,
            format_output=True,
            max_rows=max_rows
        )
        
        execution_time = time.time() - start_time
        
        # Enhance result with additional metadata
        result.update({
            "execution_time": execution_time,
            "query_used": query,
            "timestamp": time.time()
        })
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Query execution failed: {str(e)}",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "formatted": "",
            "execution_time": time.time() - start_time,
            "query_used": query,
            "timestamp": time.time()
        }


@tool
def validate_query_syntax(query: str) -> Dict[str, Any]:
    """
    Validate SQL query syntax and check for common issues.
    
    Args:
        query: SQL query to validate
        
    Returns:
        Dictionary with validation results and suggestions
    """
    validation_result = {
        "valid": True,
        "issues": [],
        "suggestions": [],
        "query_type": "unknown"
    }
    
    try:
        # Basic syntax checks
        query_clean = query.strip().rstrip(';')
        
        # Determine query type
        if query_clean.upper().startswith('SELECT'):
            validation_result["query_type"] = "select"
        elif query_clean.upper().startswith('WITH'):
            validation_result["query_type"] = "cte"
        else:
            validation_result["issues"].append("Only SELECT queries are supported")
            validation_result["valid"] = False
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        for keyword in dangerous_keywords:
            if keyword in query.upper():
                validation_result["issues"].append(f"Dangerous keyword '{keyword}' detected")
                validation_result["valid"] = False
        
        # Check for basic SQL syntax issues
        if not _check_parentheses_balance(query):
            validation_result["issues"].append("Unbalanced parentheses")
            validation_result["valid"] = False
        
        # Check for common tennis database table references
        tables_views = ['players', 'tennis_matches', 'player_match_stats', 'head_to_head', 'surface_performance']
        has_table_ref = any(table in query.lower() for table in tables_views)
        
        if not has_table_ref:
            validation_result["suggestions"].append(
                f"Consider using one of these tennis tables: {', '.join(tables_views)}"
            )
        
        # Performance suggestions
        if 'LIMIT' not in query.upper() and 'COUNT' not in query.upper():
            validation_result["suggestions"].append(
                "Consider adding LIMIT clause for large result sets"
            )
        
        if 'ORDER BY' in query.upper() and 'LIMIT' not in query.upper():
            validation_result["suggestions"].append(
                "ORDER BY without LIMIT may be slow on large datasets"
            )
        
    except Exception as e:
        validation_result["valid"] = False
        validation_result["issues"].append(f"Validation error: {str(e)}")
    
    return validation_result


@tool
def explain_query_results(results: Dict[str, Any], query_context: str = "") -> Dict[str, Any]:
    """
    Provide detailed explanation and analysis of SQL query results.
    
    Args:
        results: Results dictionary from execute_sql_query
        query_context: Additional context about what the query was trying to find
        
    Returns:
        Dictionary with detailed analysis and insights
    """
    if not results.get("success", False):
        return {
            "explanation": f"Query failed: {results.get('error', 'Unknown error')}",
            "insights": [],
            "statistical_summary": {},
            "recommendations": []
        }
    
    rows = results.get("rows", [])
    columns = results.get("columns", [])
    row_count = results.get("row_count", 0)
    
    explanation = {
        "explanation": f"Query returned {row_count} rows with {len(columns)} columns",
        "insights": [],
        "statistical_summary": {},
        "recommendations": []
    }
    
    if row_count == 0:
        explanation["explanation"] = "No results found for this query"
        explanation["recommendations"].append("Try broadening search criteria or checking player name spelling")
        return explanation
    
    # Analyze columns and data types
    explanation["statistical_summary"] = {
        "total_rows": row_count,
        "columns": columns,
        "sample_data": rows[:3] if rows else []
    }
    
    # Generate insights based on column names and data patterns
    insights = []
    
    # Check for tennis-specific patterns
    if any('rank' in col.lower() for col in columns):
        rank_cols = [i for i, col in enumerate(columns) if 'rank' in col.lower()]
        for rank_col_idx in rank_cols:
            rank_values = [row[rank_col_idx] for row in rows if row[rank_col_idx] is not None]
            if rank_values:
                min_rank = min(rank_values)
                max_rank = max(rank_values)
                insights.append(f"Rankings range from #{min_rank} to #{max_rank} (lower is better)")
    
    if any('win' in col.lower() and 'percentage' in col.lower() for col in columns):
        win_pct_cols = [i for i, col in enumerate(columns) if 'win' in col.lower() and 'percentage' in col.lower()]
        for pct_col_idx in win_pct_cols:
            pct_values = [row[pct_col_idx] for row in rows if row[pct_col_idx] is not None]
            if pct_values:
                avg_pct = sum(pct_values) / len(pct_values)
                insights.append(f"Average win percentage: {avg_pct:.1f}%")
    
    if any('surface' in col.lower() for col in columns):
        surface_col_idx = next(i for i, col in enumerate(columns) if 'surface' in col.lower())
        surfaces = [row[surface_col_idx] for row in rows]
        surface_counts = {}
        for surface in surfaces:
            surface_counts[surface] = surface_counts.get(surface, 0) + 1
        insights.append(f"Surface distribution: {surface_counts}")
    
    explanation["insights"] = insights
    
    # Generate recommendations
    recommendations = []
    
    if row_count == 1:
        recommendations.append("Single result - consider comparing with other players or broader analysis")
    elif row_count > 50:
        recommendations.append("Large result set - consider adding filters or LIMIT clause")
    
    if 'player_name' in columns:
        recommendations.append("Consider analyzing head-to-head records or surface performance for these players")
    
    if 'tournament_name' in columns:
        recommendations.append("Consider analyzing results by tournament level or surface type")
    
    explanation["recommendations"] = recommendations
    
    return explanation


@tool
def suggest_related_queries(base_query: str, results: Dict[str, Any]) -> List[str]:
    """
    Suggest related SQL queries based on the current query and its results.
    
    Args:
        base_query: The original SQL query
        results: Results from the base query
        
    Returns:
        List of suggested related queries
    """
    suggestions = []
    
    if not results.get("success", False):
        return ["-- Fix the current query first before exploring related queries"]
    
    columns = results.get("columns", [])
    rows = results.get("rows", [])
    
    # Extract player names if present
    player_names = []
    if 'player_name' in columns:
        player_col_idx = columns.index('player_name')
        player_names = [row[player_col_idx] for row in rows[:5]]  # Limit to first 5
    elif 'winner_name' in columns or 'loser_name' in columns:
        winner_idx = columns.index('winner_name') if 'winner_name' in columns else None
        loser_idx = columns.index('loser_name') if 'loser_name' in columns else None
        for row in rows[:5]:
            if winner_idx is not None:
                player_names.append(row[winner_idx])
            if loser_idx is not None:
                player_names.append(row[loser_idx])
        player_names = list(set(player_names))[:5]  # Remove duplicates, limit to 5
    
    # Generate player-specific suggestions
    if player_names:
        first_player = player_names[0]
        suggestions.append(
            f"-- Head-to-head analysis\n"
            f"SELECT * FROM head_to_head WHERE player1 = '{first_player}' OR player2 = '{first_player}' "
            f"ORDER BY h2h_matches DESC;"
        )
        
        suggestions.append(
            f"-- Surface performance for {first_player}\n"
            f"SELECT * FROM surface_performance WHERE player_name = '{first_player}' "
            f"ORDER BY win_percentage DESC;"
        )
        
        if len(player_names) > 1:
            second_player = player_names[1]
            suggestions.append(
                f"-- Direct comparison\n"
                f"SELECT player_name, total_matches, total_wins, win_percentage "
                f"FROM player_match_stats "
                f"WHERE player_name IN ('{first_player}', '{second_player}');"
            )
    
    # Tournament-based suggestions
    if 'tournament_name' in columns:
        tournament_col_idx = columns.index('tournament_name')
        tournaments = list(set([row[tournament_col_idx] for row in rows[:3]]))
        if tournaments:
            first_tournament = tournaments[0]
            suggestions.append(
                f"-- Tournament winners in {first_tournament}\n"
                f"SELECT match_date, winner_name, tournament_round "
                f"FROM tennis_matches "
                f"WHERE tournament_name = '{first_tournament}' AND tournament_round = 'Final' "
                f"ORDER BY match_date DESC;"
            )
    
    # Surface-based suggestions
    if 'surface_type' in columns:
        surface_col_idx = columns.index('surface_type')
        surfaces = list(set([row[surface_col_idx] for row in rows[:3]]))
        if surfaces:
            first_surface = surfaces[0]
            suggestions.append(
                f"-- Best performers on {first_surface}\n"
                f"SELECT player_name, win_percentage, matches_played "
                f"FROM surface_performance "
                f"WHERE surface_type = '{first_surface}' AND matches_played >= 10 "
                f"ORDER BY win_percentage DESC LIMIT 10;"
            )
    
    # Ranking-based suggestions
    if any('rank' in col.lower() for col in columns):
        suggestions.append(
            "-- Recent upsets (lower ranked beating higher ranked)\n"
            "SELECT match_date, winner_name, winner_rank, loser_name, loser_rank, "
            "(loser_rank - winner_rank) as rank_difference "
            "FROM tennis_matches "
            "WHERE winner_rank > loser_rank AND winner_rank IS NOT NULL "
            "ORDER BY rank_difference DESC LIMIT 10;"
        )
    
    # General tennis analysis suggestions
    if not suggestions:
        suggestions.extend([
            "-- Top players by wins\n"
            "SELECT player_name, total_wins, win_percentage "
            "FROM player_match_stats "
            "ORDER BY total_wins DESC LIMIT 10;",
            
            "-- Recent Grand Slam winners\n"
            "SELECT match_date, tournament_name, winner_name "
            "FROM tennis_matches "
            "WHERE tournament_level = 'Grand Slam' AND tournament_round = 'Final' "
            "ORDER BY match_date DESC LIMIT 10;",
            
            "-- Surface distribution analysis\n"
            "SELECT surface_type, COUNT(*) as match_count "
            "FROM tennis_matches "
            "GROUP BY surface_type "
            "ORDER BY match_count DESC;"
        ])
    
    return suggestions[:5]  # Limit to 5 suggestions


def _check_parentheses_balance(query: str) -> bool:
    """Check if parentheses are balanced in the query."""
    count = 0
    for char in query:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
        if count < 0:  # More closing than opening
            return False
    return count == 0  # Should be zero if balanced


def _extract_table_names(query: str) -> List[str]:
    """Extract table names from SQL query using regex."""
    # Simple regex to find table names after FROM and JOIN keywords
    from_pattern = r'\bFROM\s+(\w+)'
    join_pattern = r'\bJOIN\s+(\w+)'
    
    tables = []
    tables.extend(re.findall(from_pattern, query, re.IGNORECASE))
    tables.extend(re.findall(join_pattern, query, re.IGNORECASE))
    
    return list(set(tables))  # Remove duplicates 