#!/usr/bin/env python3
"""
Focused debug script for Text2SQL evaluation with tennis database queries.
Tests only SQL database queries from 2023-2025 timeframe.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))
os.chdir(current_dir)

try:
    from judgeval import JudgmentClient
    from judgeval.data import Example
    from judgeval.scorers import Text2SQLScorer
    from tennis_agents import TennisIntelligenceSystem, create_session_id
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def test_sql_queries():
    """Test specific SQL database queries to debug Text2SQL evaluation."""
    
    print("🔍 DEBUGGING TEXT2SQL EVALUATION")
    print("=" * 50)
    
    # Initialize tennis system with debug mode
    tennis_system = TennisIntelligenceSystem(debug=True)
    session_id = create_session_id()
    
    # Initialize JudgeVal client
    client = JudgmentClient()
    
    # Test cases that should use SQL database (2023-2025 data)
    sql_test_cases = [
        {
            "query": "Who won the French Open in 2023?",
            "expected_sql_pattern": "SELECT.*winner.*FROM.*matches.*WHERE.*tournament.*French Open.*2023"
        },
        {
            "query": "Show me Novak Djokovic's matches in 2024",
            "expected_sql_pattern": "SELECT.*FROM.*matches.*WHERE.*player.*Djokovic.*2024"
        },
        {
            "query": "What was Rafael Nadal's ranking in 2023?",
            "expected_sql_pattern": "SELECT.*ranking.*FROM.*rankings.*WHERE.*player.*Nadal.*2023"
        }
    ]
    
    for i, test_case in enumerate(sql_test_cases, 1):
        print(f"\n🎾 TEST CASE {i}: {test_case['query']}")
        print("-" * 40)
        
        try:
            # Get tennis system response
            result = tennis_system.process_query(test_case['query'], session_id)
            
            print(f"✅ Tools Called: {result.get('tools_called', [])}")
            print(f"📊 Sources: {result.get('sources', [])}")
            print(f"🗃️  SQL Data Used: {result.get('sql_data_used', False)}")
            print(f"📝 Response: {result['response'][:200]}...")
            
            # Check if SQL query is available
            sql_query = result.get('sql_query')
            if sql_query:
                print(f"🔍 SQL Query Generated: {sql_query}")
            else:
                print("⚠️  No SQL query found in result")
                print(f"🔍 Available result keys: {list(result.keys())}")
                continue
            
            # Create database schema context
            schema_context = [
                """Tennis Database Schema:
                
MAIN TABLES:

1. TENNIS_MATCHES TABLE (13,303 matches from 2023-2025)
Columns:
- match_id (INTEGER PRIMARY KEY)
- tour_type (TEXT: 'ATP', 'WTA')  
- tournament_name (TEXT)
- tournament_location (TEXT)
- match_date (DATE)
- surface_type (TEXT: 'Hard', 'Clay', 'Grass', 'Carpet')
- tournament_round (TEXT)
- winner_id (INTEGER, FK to players)
- loser_id (INTEGER, FK to players)
- winner_name (TEXT) 
- loser_name (TEXT)
- winner_rank (INTEGER, lower = better)
- loser_rank (INTEGER, lower = better)
- winner_points (INTEGER, ATP/WTA points)
- loser_points (INTEGER, ATP/WTA points)
- year (INTEGER: 2023-2025)
- month (INTEGER: 1-12)
- tournament_level (TEXT: 'Grand Slam', 'ATP/WTA 1000', etc.)

2. PLAYERS TABLE (853 players)
Columns:
- player_id (INTEGER PRIMARY KEY)
- player_name (TEXT)
- normalized_name (TEXT)
- tour_type (TEXT: 'ATP', 'WTA', 'BOTH')
- total_matches (INTEGER)
- total_wins (INTEGER)
- best_ranking (INTEGER, lower = better, career best)
- highest_points (INTEGER)
- first_appearance_date (DATE)
- last_appearance_date (DATE)

IMPORTANT NOTES:
- Player names are stored in abbreviated format: "Djokovic N." not "Novak Djokovic"
- For "Novak Djokovic" use: winner_name = 'Djokovic N.' OR loser_name = 'Djokovic N.'
- Ranking queries for specific dates: Use tennis_matches table with winner_rank/loser_rank
- best_ranking in players table is career-best, not time-specific

VIEWS:
- player_match_stats: Player statistics with win percentages
- head_to_head: Head-to-head records between players  
- surface_performance: Player performance by court surface

Date range: 2023-01-01 to 2025-06-28"""
            ]
            
            # Create example for Text2SQL evaluation
            example = Example(
                input=test_case['query'],
                actual_output=sql_query,
                context=schema_context
            )
            
            print(f"\n📋 Text2SQL Evaluation Input:")
            print(f"   Input: {example.input}")
            print(f"   SQL Query: {example.actual_output}")
            print(f"   Schema Context: {len(schema_context)} schema descriptions")
            
            # Run Text2SQL evaluation
            print(f"\n🔍 Running Text2SQL Scorer...")
            results = client.run_evaluation(
                examples=[example],
                scorers=[Text2SQLScorer],
                model="gpt-4o",
                project_name="text2sql_debug",
                eval_run_name=f"debug_test_{i}",
                override=True
            )
            
            if results and len(results) > 0:
                result_obj = results[0]
                print(f"✅ Text2SQL Result: Success = {result_obj.success}")
                
                if hasattr(result_obj, 'scorers_data') and result_obj.scorers_data:
                    for scorer_data in result_obj.scorers_data:
                        print(f"   📊 Score: {scorer_data.score}")
                        print(f"   🎯 Threshold: {scorer_data.threshold}")
                        print(f"   ✅ Success: {scorer_data.success}")
                        if scorer_data.reason:
                            print(f"   💬 Reason: {scorer_data.reason}")
                        if scorer_data.error:
                            print(f"   ❌ Error: {scorer_data.error}")
                else:
                    print("⚠️  No scorer data available")
            else:
                print("❌ No evaluation results returned")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)

def main():
    """Main debug function."""
    # Check environment variables
    if not os.getenv('JUDGMENT_API_KEY'):
        print("❌ JUDGMENT_API_KEY not set")
        return
    
    if not os.getenv('JUDGMENT_ORG_ID'):
        print("❌ JUDGMENT_ORG_ID not set")
        return
    
    test_sql_queries()

if __name__ == "__main__":
    main() 