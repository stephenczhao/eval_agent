#!/usr/bin/env python3
"""
Tennis Database SQL Query Runner
================================

A utility to run SQL queries against the tennis matches database.
Designed for LLM agents and interactive use.

Function API for LLM agents:
    from run_sql import run_sql_query
    
    result = run_sql_query("SELECT * FROM players LIMIT 5")
    print(result)

Command Line Usage:
    python run_sql.py                          # Interactive mode
    python run_sql.py "SELECT * FROM players LIMIT 5"  # Single query
    python run_sql.py --file queries.sql       # Run queries from file

Demo Queries:
    # Top ATP players by wins
    SELECT player_name, total_wins, win_percentage 
    FROM player_match_stats 
    WHERE tour_type = 'ATP' 
    ORDER BY total_wins DESC LIMIT 10;

    # Head-to-head between specific players
    SELECT * FROM head_to_head 
    WHERE (player1 = 'Djokovic N.' AND player2 = 'Sinner J.') 
    OR (player1 = 'Sinner J.' AND player2 = 'Djokovic N.');

    # Most successful players on each surface
    SELECT surface_type, player_name, win_percentage, matches_played
    FROM surface_performance 
    WHERE matches_played >= 20
    ORDER BY surface_type, win_percentage DESC;

    # Recent upsets (lower ranked beating higher ranked)
    SELECT match_date, winner_name, winner_rank, loser_name, loser_rank, 
           tournament_name, (loser_rank - winner_rank) as rank_difference
    FROM tennis_matches 
    WHERE winner_rank > loser_rank 
    AND winner_rank IS NOT NULL AND loser_rank IS NOT NULL
    ORDER BY rank_difference DESC LIMIT 10;

    # Tournament winners in 2024
    SELECT tournament_name, tournament_location, match_date, 
           tour_type, winner_name
    FROM tennis_matches 
    WHERE year = 2024 AND tournament_round = 'Final'
    ORDER BY match_date;

    # Players with most wins against top 10 players
    SELECT winner_name, COUNT(*) as wins_vs_top10
    FROM tennis_matches 
    WHERE loser_rank <= 10 AND loser_rank IS NOT NULL
    GROUP BY winner_name 
    ORDER BY wins_vs_top10 DESC LIMIT 15;

    # Grand Slam performance by player
    SELECT winner_name, tournament_name, COUNT(*) as titles
    FROM tennis_matches 
    WHERE tournament_level = 'Grand Slam' AND tournament_round = 'Final'
    GROUP BY winner_name, tournament_name
    ORDER BY winner_name, titles DESC;

    # Monthly match distribution
    SELECT 
        CASE month 
            WHEN 1 THEN 'January' WHEN 2 THEN 'February' WHEN 3 THEN 'March'
            WHEN 4 THEN 'April' WHEN 5 THEN 'May' WHEN 6 THEN 'June'
            WHEN 7 THEN 'July' WHEN 8 THEN 'August' WHEN 9 THEN 'September'
            WHEN 10 THEN 'October' WHEN 11 THEN 'November' WHEN 12 THEN 'December'
        END as month_name,
        COUNT(*) as matches
    FROM tennis_matches 
    GROUP BY month 
    ORDER BY month;

    # Cross-tour analysis (players in both ATP and WTA)
    SELECT p.player_name, p.total_matches, p.total_wins,
           COUNT(DISTINCT m.tour_type) as tours_played
    FROM players p
    JOIN tennis_matches m ON (m.winner_id = p.player_id OR m.loser_id = p.player_id)
    WHERE p.tour_type = 'BOTH'
    GROUP BY p.player_id
    ORDER BY p.total_matches DESC;

    # Longest winning streaks (approximate - consecutive wins)
    WITH player_matches AS (
        SELECT winner_name as player, match_date, 'W' as result
        FROM tennis_matches
        UNION ALL
        SELECT loser_name as player, match_date, 'L' as result
        FROM tennis_matches
    )
    SELECT player, COUNT(*) as consecutive_wins
    FROM (
        SELECT player, match_date, result,
               ROW_NUMBER() OVER (PARTITION BY player ORDER BY match_date) -
               ROW_NUMBER() OVER (PARTITION BY player, result ORDER BY match_date) as grp
        FROM player_matches
    ) grouped
    WHERE result = 'W'
    GROUP BY player, grp
    ORDER BY consecutive_wins DESC
    LIMIT 10;
"""

import sqlite3
import sys
import argparse
from pathlib import Path
import csv
from typing import List, Tuple, Any, Optional, Dict, Union

def run_sql_query(query: str, db_path: str = "tennis_data/tennis_matches.db", 
                  format_output: bool = True, max_rows: int = 100) -> Dict[str, Any]:
    """
    Execute a SQL query against the tennis matches database and return structured results.
    
    This function provides a clean interface for LLM agents to execute SQL queries
    against the tennis database and receive structured, JSON-serializable results.
    
    The tennis database contains:
    - tennis_matches: Match details (13,303 matches from 2023-2025)
    - players: Player information and career statistics (853 players)
    - player_match_stats: View with aggregated player statistics
    - head_to_head: View with head-to-head matchup records
    - surface_performance: View with player performance by court surface
    
    Args:
        query (str): SQL query to execute against the tennis database
        db_path (str, optional): Path to SQLite database file. Defaults to "tennis_data/tennis_matches.db"
        format_output (bool, optional): Whether to include a formatted table string for display. 
                                      Defaults to True.
        max_rows (int, optional): Maximum number of rows to return. Use 0 for unlimited. 
                                Defaults to 100.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): True if query executed successfully, False otherwise
            - columns (List[str]): List of column names from the query result
            - rows (List[List[Any]]): Query results as a list of lists (JSON-serializable)
            - row_count (int): Total number of rows returned by the query
            - formatted (str): Human-readable formatted table string (if format_output=True)
            - error (Optional[str]): Error message if query failed, None if successful
    
    Raises:
        No exceptions are raised. All errors are captured and returned in the result dict.
    
    Examples:
        # Basic query
        result = run_sql_query("SELECT COUNT(*) FROM tennis_matches")
        if result['success']:
            print(f"Total matches: {result['rows'][0][0]}")
        
        # Player statistics
        result = run_sql_query('''
            SELECT player_name, total_wins, win_percentage 
            FROM player_match_stats 
            WHERE tour_type = 'ATP' 
            ORDER BY total_wins DESC 
            LIMIT 5
        ''')
        if result['success']:
            print(result['formatted'])  # Pretty table output
            
        # Error handling
        result = run_sql_query("SELECT * FROM nonexistent_table")
        if not result['success']:
            print(f"Query failed: {result['error']}")
    
    Note:
        The function automatically handles database connection management and ensures
        all results are JSON-serializable for easy integration with LLM agents and APIs.
    """
    try:
        # Check if database exists
        db_file = Path(db_path)
        if not db_file.exists():
            return {
                'success': False,
                'error': f"Database file not found: {db_path}",
                'columns': [],
                'rows': [],
                'row_count': 0,
                'formatted': ''
            }
        
        # Execute query with proper connection management
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Get rows (limit to max_rows)
            all_rows = cursor.fetchall()
            limited_rows = all_rows[:max_rows] if max_rows > 0 else all_rows
            
            # Convert to list of lists for JSON serialization
            rows = [list(row) for row in limited_rows]
            
        finally:
            if conn:
                conn.close()
        
        # Format output if requested
        formatted_output = ""
        if format_output and columns and rows:
            formatted_output = _format_query_results(columns, limited_rows, len(all_rows), max_rows)
        
        return {
            'success': True,
            'columns': columns,
            'rows': rows,
            'row_count': len(all_rows),
            'formatted': formatted_output,
            'error': None
        }
        
    except sqlite3.Error as e:
        return {
            'success': False,
            'error': f"SQL Error: {str(e)}",
            'columns': [],
            'rows': [],
            'row_count': 0,
            'formatted': ''
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Error: {str(e)}",
            'columns': [],
            'rows': [],
            'row_count': 0,
            'formatted': ''
        }

def _format_query_results(columns: List[str], rows: List[Tuple[Any, ...]], 
                         total_rows: int, max_rows: int, max_width: int = 120) -> str:
    """Format query results as a nice table string."""
    if not columns or not rows:
        return "No results found."
    
    # Calculate column widths
    col_widths = []
    for i, col in enumerate(columns):
        max_len = len(str(col))
        for row in rows:
            if i < len(row):
                max_len = max(max_len, len(str(row[i])))
        # Limit individual column width
        col_widths.append(min(max_len, max_width // len(columns)))
    
    # Create format string
    format_str = " | ".join(f"{{:<{width}}}" for width in col_widths)
    
    # Build table
    result = []
    
    # Header
    header = format_str.format(*[col[:w] for col, w in zip(columns, col_widths)])
    result.append(header)
    result.append("-" * len(header))
    
    # Rows
    for row in rows:
        formatted_row = []
        for i, (val, width) in enumerate(zip(row, col_widths)):
            val_str = str(val) if val is not None else "NULL"
            formatted_row.append(val_str[:width])
        result.append(format_str.format(*formatted_row))
    
    # Add summary
    if max_rows > 0 and total_rows > max_rows:
        result.append(f"\n... ({total_rows - max_rows} more rows not shown)")
    result.append(f"\nTotal rows: {total_rows}")
    
    return "\n".join(result)

class TennisDBRunner:
    """Tennis database query runner with formatting and export capabilities."""
    
    def __init__(self, db_path: str = "tennis_data/tennis_matches.db"):
        """Initialize with database path."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def execute_query(self, query: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
        """Execute SQL query and return column names and rows."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Get all rows
            rows = cursor.fetchall()
            
            conn.close()
            return columns, rows
            
        except sqlite3.Error as e:
            raise sqlite3.Error(f"SQL Error: {e}")
        except Exception as e:
            raise Exception(f"Error executing query: {e}")
    
    def format_results(self, columns: List[str], rows: List[Tuple[Any, ...]], 
                      max_width: int = 100) -> str:
        """Format query results as a nice table."""
        if not columns or not rows:
            return "No results found."
        
        # Calculate column widths
        col_widths = []
        for i, col in enumerate(columns):
            max_len = len(str(col))
            for row in rows:
                if i < len(row):
                    max_len = max(max_len, len(str(row[i])))
            col_widths.append(min(max_len, max_width // len(columns)))
        
        # Create format string
        format_str = " | ".join(f"{{:<{width}}}" for width in col_widths)
        
        # Build table
        result = []
        
        # Header
        header = format_str.format(*[col[:w] for col, w in zip(columns, col_widths)])
        result.append(header)
        result.append("-" * len(header))
        
        # Rows
        for row in rows[:50]:  # Limit to first 50 rows for display
            formatted_row = []
            for i, (val, width) in enumerate(zip(row, col_widths)):
                val_str = str(val) if val is not None else "NULL"
                formatted_row.append(val_str[:width])
            result.append(format_str.format(*formatted_row))
        
        # Add truncation notice if needed
        if len(rows) > 50:
            result.append(f"\n... ({len(rows) - 50} more rows truncated for display)")
        
        result.append(f"\nTotal rows: {len(rows)}")
        return "\n".join(result)
    
    def export_csv(self, columns: List[str], rows: List[Tuple[Any, ...]], 
                   filename: str) -> None:
        """Export results to CSV file."""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            writer.writerows(rows)
        print(f"Results exported to {filename}")
    
    def run_query(self, query: str, export_csv: Optional[str] = None, 
                  quiet: bool = False) -> None:
        """Run a single query and display/export results."""
        try:
            if not quiet:
                print(f"Executing query:\n{query}\n")
            
            # Use the main function
            result = run_sql_query(query, str(self.db_path), format_output=True, max_rows=0)
            
            if not result['success']:
                print(f"Error: {result['error']}")
                return
            
            columns = result['columns']
            rows = [tuple(row) for row in result['rows']]  # Convert back to tuples for CSV export
            
            if export_csv:
                self.export_csv(columns, rows, export_csv)
            
            if not quiet:
                print(result['formatted'])
                
        except Exception as e:
            print(f"Error: {e}")
    
    def interactive_mode(self) -> None:
        """Run in interactive mode."""
        print("Tennis Database Query Runner")
        print("============================")
        print(f"Connected to: {self.db_path}")
        print("\nType SQL queries (end with semicolon)")
        print("Special commands:")
        print("  .help    - Show this help")
        print("  .tables  - List all tables")
        print("  .schema  - Show table schemas")
        print("  .demo    - Run demo queries")
        print("  .exit    - Exit")
        print()
        
        query_buffer = []
        
        while True:
            try:
                if query_buffer:
                    prompt = "... "
                else:
                    prompt = "sql> "
                
                line = input(prompt).strip()
                
                # Handle special commands
                if line.startswith('.'):
                    if line == '.exit':
                        break
                    elif line == '.help':
                        self.show_help()
                    elif line == '.tables':
                        self.show_tables()
                    elif line == '.schema':
                        self.show_schema()
                    elif line == '.demo':
                        self.run_demo_queries()
                    else:
                        print(f"Unknown command: {line}")
                    continue
                
                # Build query
                query_buffer.append(line)
                query = " ".join(query_buffer)
                
                # Execute if query ends with semicolon
                if query.rstrip().endswith(';'):
                    query = query.rstrip()[:-1]  # Remove semicolon
                    self.run_query(query)
                    query_buffer = []
                    print()
                    
            except KeyboardInterrupt:
                print("\nQuery cancelled.")
                query_buffer = []
            except EOFError:
                break
        
        print("Goodbye!")
    
    def show_help(self) -> None:
        """Show help information."""
        print("\nTennis Database Schema:")
        print("- players: Player information and statistics")
        print("- tennis_matches: Match details and results")
        print("- player_match_stats: View with player statistics")
        print("- head_to_head: View with head-to-head records")
        print("- surface_performance: View with surface-specific stats")
        print("\nExample queries available with .demo command")
    
    def show_tables(self) -> None:
        """Show all tables in the database."""
        query = "SELECT name FROM sqlite_master WHERE type='table' OR type='view' ORDER BY name"
        columns, rows = self.execute_query(query)
        print("\nTables and Views:")
        for row in rows:
            print(f"  {row[0]}")
    
    def show_schema(self) -> None:
        """Show schema for all tables."""
        # Get tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        _, tables = self.execute_query(tables_query)
        
        for (table_name,) in tables:
            print(f"\n{table_name}:")
            schema_query = f"PRAGMA table_info({table_name})"
            columns, rows = self.execute_query(schema_query)
            for row in rows:
                nullable = "NULL" if row[3] == 0 else "NOT NULL"
                pk = " PRIMARY KEY" if row[5] == 1 else ""
                print(f"  {row[1]} {row[2]} {nullable}{pk}")
    
    def run_demo_queries(self) -> None:
        """Run demonstration queries."""
        demo_queries = [
            ("Top 5 ATP Players by Wins", """
                SELECT player_name, total_wins, win_percentage 
                FROM player_match_stats 
                WHERE tour_type = 'ATP' 
                ORDER BY total_wins DESC LIMIT 5
            """),
            
            ("Surface Distribution", """
                SELECT surface_type, COUNT(*) as matches 
                FROM tennis_matches 
                GROUP BY surface_type 
                ORDER BY matches DESC
            """),
            
            ("Recent Tournament Winners", """
                SELECT match_date, tournament_name, winner_name, tour_type
                FROM tennis_matches 
                WHERE tournament_round = 'Final' 
                ORDER BY match_date DESC LIMIT 10
            """),
            
            ("Head-to-Head: Most Played Rivalries", """
                SELECT player1, player2, h2h_matches, 
                       player1_wins, player2_wins
                FROM head_to_head 
                ORDER BY h2h_matches DESC LIMIT 5
            """),
            
            ("Best Clay Court Players (min 20 matches)", """
                SELECT player_name, matches_played, win_percentage
                FROM surface_performance 
                WHERE surface_type = 'Clay' AND matches_played >= 20
                ORDER BY win_percentage DESC LIMIT 10
            """)
        ]
        
        for title, query in demo_queries:
            print(f"\n{'='*50}")
            print(f"Demo: {title}")
            print('='*50)
            self.run_query(query.strip(), quiet=False)
            input("\nPress Enter to continue...")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tennis Database SQL Query Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'query', 
        nargs='?', 
        help='SQL query to execute (if not provided, enters interactive mode)'
    )
    
    parser.add_argument(
        '--db', 
        default='tennis_data/tennis_matches.db',
        help='Path to database file (default: tennis_data/tennis_matches.db)'
    )
    
    parser.add_argument(
        '--file', 
        help='Read queries from file (one per line)'
    )
    
    parser.add_argument(
        '--csv', 
        help='Export results to CSV file'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Suppress query output (useful with --csv)'
    )
    
    args = parser.parse_args()
    
    try:
        runner = TennisDBRunner(args.db)
        
        if args.file:
            # Read queries from file
            with open(args.file, 'r') as f:
                queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            for i, query in enumerate(queries):
                if not args.quiet:
                    print(f"\n--- Query {i+1} ---")
                csv_file = f"{args.csv}_{i+1}.csv" if args.csv else None
                runner.run_query(query, csv_file, args.quiet)
                
        elif args.query:
            # Single query mode
            runner.run_query(args.query, args.csv, args.quiet)
        else:
            # Interactive mode
            runner.interactive_mode()
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # If called directly, run command line interface
    # run the following to test the database
    if len(sys.argv) == 1:
        # No arguments - check if we should run a quick test
        import os
        if os.path.exists("tennis_data/tennis_matches.db"):
            print("Testing run_sql_query function:")
            print("=" * 50)
            
            # Quick test
            result = run_sql_query("SELECT COUNT(*) as total_matches FROM tennis_matches")
            if result['success']:
                print("✓ Function working correctly!")
                print(result['formatted'])
                print("\nTo use interactively: python src/utils/run_sql.py")
                print("To run specific query: python src/utils/run_sql.py \"YOUR_SQL_QUERY\"")
                print("\nFor LLM agents, import the function:")
                print("  from run_sql import run_sql_query")
                print("  result = run_sql_query('SELECT * FROM players LIMIT 5')")
            else:
                print(f"✗ Error: {result['error']}")
        else:
            print("Database not found. Run create_sql_db.py first.")
    else:
        main()