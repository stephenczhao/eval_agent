"""
Tennis Intelligence System Evaluation with JudgeVal
==================================================

This module integrates JudgeVal evaluation capabilities with the tennis intelligence system
to provide comprehensive performance monitoring and evaluation.

Features:
- Automated evaluation of tennis system responses
- Multiple evaluation metrics (faithfulness, relevancy, hallucination detection, etc.)
- Test case generation for various tennis query types
- Performance benchmarking and reporting
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# hide warnings
import warnings
warnings.filterwarnings("ignore")

# Add src directory to Python path for proper imports
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Change directory to project root for relative file paths
os.chdir(current_dir)


from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    FaithfulnessScorer,
    AnswerRelevancyScorer,
    AnswerCorrectnessScorer,
    InstructionAdherenceScorer,
    ToolOrderScorer,
    ToolDependencyScorer,
    Text2SQLScorer,
    ClassifierScorer,
)
# NEW: Add LangGraph integration for trace capture
from judgeval.integrations.langgraph import JudgevalCallbackHandler
from judgeval.common.tracer import Tracer
from tennis_agents import TennisAgentSystem, create_session_id
from config.settings import TennisConfig


@dataclass
class TennisEvaluationResult:
    """Container for tennis system evaluation results."""
    query: str
    response: str
    confidence: float
    sources: List[str]
    tools_called: List[str]
    processing_time: float
    evaluation_scores: Dict[str, Any]
    success: bool
    errors: List[str]


class TennisEvaluationSuite:
    """
    Comprehensive evaluation suite for the Tennis Intelligence System using JudgeVal.
    
    This class provides structured evaluation of the tennis system across multiple
    dimensions including faithfulness, relevancy, hallucination detection, and tool usage.
    """
    
    def __init__(self, 
                 project_name: str = "tennis_intelligence_eval",
                 debug: bool = False):
        """
        Initialize the tennis evaluation suite.
        
        Args:
            project_name: JudgeVal project name for organizing evaluations
            debug: Enable debug mode for detailed logging
        """
        self.debug = debug
        self.project_name = project_name
        
        # Initialize tennis system
        self.tennis_system = TennisAgentSystem(debug=debug)
        self.session_id = create_session_id()
        
        # NEW: Initialize JudgeVal tracer for LangGraph integration
        self.tracer = Tracer(
            project_name=project_name,
            enable_monitoring=True,
            enable_evaluations=True
        )
        
        # NEW: Create callback handler for trace capture
        self.callback_handler = JudgevalCallbackHandler(self.tracer)
        
        # Initialize JudgeVal client
        self.judgment_client = JudgmentClient()
        if debug:
            print("✅ JudgeVal client initialized successfully")
            print("✅ LangGraph trace capture enabled")
        
        # Define evaluation scorers with appropriate thresholds
        self.scorers = [
            FaithfulnessScorer(threshold=0.7),          # High threshold for factual accuracy
            AnswerRelevancyScorer(threshold=0.8),       # High threshold for relevance
            InstructionAdherenceScorer(threshold=0.7),   # Check if instructions are followed
            ToolOrderScorer(threshold=0.8),             # Evaluate tool usage order
            ToolDependencyScorer(threshold=0.8),        # Check tool dependencies
            Text2SQLScorer,                             # Evaluate SQL query correctness (for database queries)
            AnswerCorrectnessScorer,
            
            # NOTE: Advanced scorers (HallucinationScorer, DerailmentScorer, ExecutionOrderScorer)
            # are available in judgeval >= 0.0.54. Upgrade with: pip install --upgrade judgeval
        ]
        
        # Add custom tennis expertise classifier
        self.tennis_expertise_scorer = ClassifierScorer(
            name="Tennis Expertise",
            slug="tennis-expertise-eval",
            threshold=0.7,
            conversation=[
                {
                    "role": "system", 
                    "content": """You are evaluating whether a tennis-related response demonstrates good tennis knowledge and expertise. 

Consider:
- Accuracy of tennis facts, rules, and statistics
- Proper use of tennis terminology
- Contextual understanding of tennis history and current events
- Quality of insights and analysis

Question: {{input}}
Response: {{actual_output}}

Does this response demonstrate good tennis expertise? Answer Y for yes, N for no."""
                }
            ],
            options={"Y": 1.0, "N": 0.0}
        )
        
        # Add tennis expertise scorer to the list
        self.scorers.append(self.tennis_expertise_scorer)
        
        if debug:
            print(f"🎾 Tennis Evaluation Suite initialized")
            print(f"📊 Project: {project_name}")
            print(f"🔍 Scorers: {[scorer.__class__.__name__ for scorer in self.scorers]}")
            print(f"📋 Capabilities: Trace capture, Tennis expertise evaluation")
            print(f"💡 To enable advanced scorers (Hallucination, Derailment, ExecutionOrder), upgrade:")
            print(f"   pip install --upgrade judgeval")
    
    def create_tennis_test_cases(self) -> List[Dict[str, Any]]:
        """
        Create comprehensive test cases for tennis system evaluation.
        
        Returns:
            List of test case dictionaries with queries and expected characteristics
        """
        test_cases = [
            # Recent data queries (should use online search)
            {
                "query": "What are the current ATP rankings as of today?",
                "expected_output": "As of July 16, 2025, the world No. 1 in men’s tennis is Jannik Sinner, sitting atop the ATP rankings with 12,030 points, while the WTA No. 1 is Aryna Sabalenka, leading with 12,420 points.",
                "expected_tools": [{"tool_name": "online_search", "parameters": None}],
                "category": "current_data",
                "expected_characteristics": {
                    "should_mention_recency": True,
                    "should_use_search": True,
                    "temporal_scope": "current"
                }
            },
        #     {
        #         "query": "Who won the latest Grand Slam tournament?",
        #         "expected_tools": [{"tool_name": "online_search", "parameters": None}],
        #         "category": "recent_events",
        #         "expected_characteristics": {
        #             "should_mention_recency": True,
        #             "should_use_search": True
        #         }
        #     },
            
        #     # Historical data queries (should use SQL database)
        #     {
        #         "query": "Who won the French Open in 2023?",
        #         "expected_tools": [{"tool_name": "query_sql_database", "parameters": None}],
        #         "category": "historical_data",
        #         "expected_characteristics": {
        #             "should_use_database": True,
        #             "temporal_scope": "2023"
        #         }
        #     },
        #     {
        #         "query": "Show me Novak Djokovic's performance in 2024",
        #         "expected_tools": [{"tool_name": "query_sql_database", "parameters": None}],
        #         "category": "player_analysis",
        #         "expected_characteristics": {
        #             "should_use_database": True,
        #             "should_include_stats": True
        #         }
        #     },
        #     {
        #         "query": "Compare Rafael Nadal and Roger Federer's head-to-head record from 2023-2024",
        #         "expected_tools": [{"tool_name": "query_sql_database", "parameters": None}],
        #         "category": "comparison_analysis",
        #         "expected_characteristics": {
        #             "should_use_database": True,
        #             "should_compare_players": True
        #         }
        #     },
            
        #     # Edge cases and temporal boundary queries
        #     {
        #         "query": "What happened in tennis in 2022?",
        #         "expected_tools": [{"tool_name": "online_search", "parameters": None}],
        #         "category": "pre_database_era",
        #         "expected_characteristics": {
        #             "should_use_search": True,
        #             "temporal_scope": "pre_database"
        #         }
        #     },
        #     {
        #         "query": "Tennis predictions for 2026",
        #         "expected_tools": [{"tool_name": "online_search", "parameters": None}], 
        #         "category": "future_queries",
        #         "expected_characteristics": {
        #             "should_use_search": True,
        #             "temporal_scope": "future"
        #         }
        #     },
            
        #     # Mixed temporal queries
        #     {
        #         "query": "How has Serena Williams' ranking changed over the years?",
        #         "expected_tools": [
        #             {"tool_name": "query_sql_database", "parameters": None},
        #             {"tool_name": "online_search", "parameters": None}
        #         ],
        #         "category": "long_term_analysis",
        #         "expected_characteristics": {
        #             "should_use_both_sources": True,
        #             "should_mention_career_span": True
        #         }
        #     },
            
        #     # Technical tennis queries
        #     {
        #         "query": "Explain the differences between clay, grass, and hard court surfaces in tennis",
        #         "expected_tools": [{"tool_name": "online_search", "parameters": None}],
        #         "category": "technical_knowledge",
        #         "expected_characteristics": {
        #             "should_be_educational": True,
        #             "should_explain_differences": True
        #         }
        #     },
            
        #     # Specific statistical queries
        #     {
        #         "query": "What was the average match duration in ATP tournaments in 2024?",
        #         "expected_tools": [{"tool_name": "query_sql_database", "parameters": None}],
        #         "category": "statistical_analysis",
        #         "expected_characteristics": {
        #             "should_use_database": True,
        #             "should_include_numbers": True
        #         }
        #     },
            
        #     # NEW: Test cases for new scorers
        #     {
        #         "query": "Tell me about the current political situation in tennis",
        #         "expected_tools": [{"tool_name": "online_search", "parameters": None}],
        #         "category": "derailment_test",
        #         "expected_characteristics": {
        #             "should_stay_tennis_focused": True,
        #             "should_not_discuss_politics": True
        #         }
        #     },
        #     {
        #         "query": "What's Roger Federer's record against aliens from Mars?", 
        #         "expected_tools": [{"tool_name": "online_search", "parameters": None}],
        #         "category": "hallucination_test",
        #         "expected_characteristics": {
        #             "should_not_make_up_facts": True,
        #             "should_acknowledge_impossibility": True
        #         }
        #     },
        #     {
        #         "query": "How many Grand Slams did Rafael Nadal win on the moon?",
        #         "expected_tools": [{"tool_name": "query_sql_database", "parameters": None}],
        #         "category": "nonsensical_test", 
        #         "expected_characteristics": {
        #             "should_clarify_impossibility": True,
        #             "should_not_hallucinate": True
        #         }
        #     },
        ]
        
        if self.debug:
            print(f"📝 Created {len(test_cases)} test cases")
            print(f"📊 Categories: {set(tc['category'] for tc in test_cases)}")
        
        return test_cases
    
    def run_tennis_evaluation(self, 
                            test_cases: Optional[List[Dict[str, Any]]] = None,
                            eval_run_name: Optional[str] = None) -> List[TennisEvaluationResult]:
        """
        Run comprehensive evaluation of the tennis system.
        
        Args:
            test_cases: Custom test cases, or None to use default cases
            eval_run_name: Name for this evaluation run
            
        Returns:
            List of evaluation results
        """
        if test_cases is None:
            test_cases = self.create_tennis_test_cases()
        
        if eval_run_name is None:
            eval_run_name = f"tennis_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.debug:
            print(f"\n🚀 Starting Tennis System Evaluation")
            print(f"📊 Eval Run: {eval_run_name}")
            print(f"📝 Test Cases: {len(test_cases)}")
        
        # Run tennis system on each test case and collect responses
        tennis_examples = []
        tennis_results = []
        
        for i, test_case in enumerate(test_cases):
            if self.debug:
                print(f"\n🎾 Running test case {i+1}/{len(test_cases)}: {test_case['category']}")
                print(f"❓ Query: {test_case['query']}")
            
            try:
                # Create fresh session for each test case to prevent memory contamination
                fresh_session_id = create_session_id()
                
                # NEW: Reset callback handler for each test case to get clean traces
                self.callback_handler.reset()
                
                # Get tennis system response with trace capture
                start_time = time.time()
                result = self.tennis_system.process_query(
                    test_case['query'], 
                    fresh_session_id,
                    callbacks=[self.callback_handler]  # NEW: Enable trace capture
                )
                processing_time = time.time() - start_time
                
                # Create retrieval context from sources and metadata
                retrieval_context = []
                if result.get('sources'):
                    retrieval_context.extend(result['sources'])
                if result.get('tools_called'):
                    retrieval_context.append(f"Tools used: {', '.join(result['tools_called'])}")
                if result.get('database_temporal_range'):
                    retrieval_context.append(f"Database range: {result['database_temporal_range']}")
                if result.get('sql_query'):
                    retrieval_context.append(f"SQL Query: {result['sql_query']}")
                
                # Create context for Text2SQL evaluation (database schema for SQL queries)
                context = None
                if 'query_sql_database' in result.get('tools_called', []):
                    context = [
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
                
                # Use SQL query as actual_output for Text2SQL evaluation if available
                actual_output_for_sql = result.get('sql_query', result['response'])
                
                # NEW: Extract trace information from callback handler
                captured_traces = self.tracer.traces
                tools_called_from_trace = []
                execution_sequence = []
                
                if captured_traces:
                    # Get the most recent trace (for this test case)
                    latest_trace = captured_traces[-1] if captured_traces else {}
                    trace_spans = latest_trace.get('trace_spans', [])
                    
                    # Extract tools and execution order from spans
                    for span in trace_spans:
                        span_data = span if isinstance(span, dict) else span
                        function_name = span_data.get('function', '')
                        
                        if 'query_sql_database' in function_name:
                            tools_called_from_trace.append('query_sql_database')
                            execution_sequence.append({'tool_name': 'query_sql_database', 'parameters': {}})
                        elif 'online_search' in function_name:
                            tools_called_from_trace.append('online_search')
                            execution_sequence.append({'tool_name': 'online_search', 'parameters': {}})
                
                # Merge tools from result and trace
                all_tools_called = list(set(result.get('tools_called', []) + tools_called_from_trace))
                
                # Create JudgeVal Example
                example = Example(
                    input=test_case['query'],
                    actual_output=actual_output_for_sql if context else result['response'],
                    retrieval_context=retrieval_context,
                    context=context,  # For Text2SQL evaluation
                    tools_called=all_tools_called,  # NEW: Enhanced with trace data
                    expected_tools=test_case.get('expected_tools', []),
                    additional_metadata={
                        'category': test_case['category'],
                        'confidence': result.get('confidence', 0.0),
                        'processing_time': processing_time,
                        'expected_characteristics': test_case.get('expected_characteristics', {}),
                        'tennis_system_metadata': {
                            'sources': result.get('sources', []),
                            'sql_data_used': result.get('sql_data_used', False),
                            'search_data_used': result.get('search_data_used', False),
                            'langgraph_used': result.get('langgraph_used', False)
                        },
                        # NEW: Add trace information for execution order evaluation
                        'execution_sequence': execution_sequence,
                        'trace_captured': len(captured_traces) > 0,
                        'trace_spans_count': len(trace_spans) if captured_traces else 0
                    }
                )
                
                tennis_examples.append(example)
                
                # Store result for later analysis
                tennis_results.append(TennisEvaluationResult(
                    query=test_case['query'],
                    response=result['response'],
                    confidence=result.get('confidence', 0.0),
                    sources=result.get('sources', []),
                    tools_called=result.get('tools_called', []),
                    processing_time=processing_time,
                    evaluation_scores={},  # Will be filled after JudgeVal evaluation
                    success=not result.get('error', False),
                    errors=[result.get('error')] if result.get('error') else []
                ))
                
                if self.debug:
                    print(f"✅ Response: {result['response'][:100]}...")
                    print(f"🔧 Tools: {all_tools_called}")  # Updated to show enhanced tools
                    print(f"⏱️  Time: {processing_time:.2f}s")
                    print(f"📊 Trace spans captured: {len(trace_spans) if captured_traces else 0}")
                    if execution_sequence:
                        print(f"🔄 Execution sequence: {[item['tool_name'] for item in execution_sequence]}")
                
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                print(f"❌ {error_msg}")
                
                # Create error example for evaluation
                example = Example(
                    input=test_case['query'],
                    actual_output=error_msg,
                    retrieval_context=[],
                    additional_metadata={
                        'category': test_case['category'],
                        'error': True,
                        'error_type': type(e).__name__,
                        'trace_captured': False
                    }
                )
                tennis_examples.append(example)
                
                # Store error result
                tennis_results.append(TennisEvaluationResult(
                    query=test_case['query'],
                    response=error_msg,
                    confidence=0.0,
                    sources=[],
                    tools_called=[],
                    processing_time=0.0,
                    evaluation_scores={},
                    success=False,
                    errors=[error_msg]
                ))
        
        # Run JudgeVal evaluation
        if self.debug:
            print(f"\n🔍 Running JudgeVal evaluation with {len(self.scorers)} scorers...")
        
        try:
            scoring_results = self.judgment_client.run_evaluation(
                examples=tennis_examples,
                scorers=self.scorers,
                model="gpt-4o",  # Use latest model for evaluation
                project_name=self.project_name,
                eval_run_name=eval_run_name,
                override=True  # Allow overwriting for iterative testing
            )
            
            if self.debug:
                print(f"✅ JudgeVal evaluation completed: {len(scoring_results)} results")
            
        except Exception as e:
            print(f"❌ JudgeVal evaluation failed: {e}")
            scoring_results = []
        
        # Merge JudgeVal scores with tennis results
        for i, tennis_result in enumerate(tennis_results):
            if i < len(scoring_results) and scoring_results[i].scorers_data:
                # Extract scores from JudgeVal results
                scores = {}
                for scorer_data in scoring_results[i].scorers_data:
                    scores[scorer_data.name] = {
                        'score': scorer_data.score,
                        'success': scorer_data.success,
                        'reason': scorer_data.reason,
                        'threshold': scorer_data.threshold
                    }
                tennis_result.evaluation_scores = scores
        
        return tennis_results
    
    def analyze_evaluation_results(self, results: List[TennisEvaluationResult]) -> Dict[str, Any]:
        """
        Analyze evaluation results and provide comprehensive insights.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary containing analysis summary
        """
        if not results:
            return {"error": "No results to analyze"}
        
        analysis = {
            "summary": {
                "total_queries": len(results),
                "successful_queries": sum(1 for r in results if r.success),
                "failed_queries": sum(1 for r in results if not r.success),
                "average_confidence": sum(r.confidence for r in results) / len(results),
                "average_processing_time": sum(r.processing_time for r in results) / len(results)
            },
            "tool_usage": {},
            "category_performance": {},
            "evaluation_scores": {},
            "trace_analysis": {},  # NEW: Trace capture insights
            "issues_found": []
        }
        
        # Analyze tool usage
        all_tools = set()
        for result in results:
            all_tools.update(result.tools_called)
        
        for tool in all_tools:
            usage_count = sum(1 for r in results if tool in r.tools_called)
            analysis["tool_usage"][tool] = {
                "count": usage_count,
                "percentage": (usage_count / len(results)) * 100
            }
        
        # Analyze performance by category
        categories = set()
        for result in results:
            # Extract category from evaluation scores metadata if available
            category = "unknown"
            if result.evaluation_scores and isinstance(result.evaluation_scores, dict):
                # Try to extract category from the first scorer's metadata
                for scorer_name, scorer_data in result.evaluation_scores.items():
                    if isinstance(scorer_data, dict) and 'category' in str(scorer_data):
                        category = "extracted_from_scorer"
                        break
            categories.add(category)
        
        for category in categories:
            category_results = [r for r in results if category == "unknown"]  # Simplified for now
            if category_results:
                analysis["category_performance"][category] = {
                    "total": len(category_results),
                    "successful": sum(1 for r in category_results if r.success),
                    "average_confidence": sum(r.confidence for r in category_results) / len(category_results)
                }
        
        # Analyze evaluation scores
        if results and results[0].evaluation_scores:
            scorer_names = set()
            for result in results:
                if result.evaluation_scores:
                    scorer_names.update(result.evaluation_scores.keys())
            
            for scorer_name in scorer_names:
                scores = []
                successes = []
                for result in results:
                    if result.evaluation_scores and scorer_name in result.evaluation_scores:
                        scorer_data = result.evaluation_scores[scorer_name]
                        if isinstance(scorer_data, dict):
                            if 'score' in scorer_data and scorer_data['score'] is not None:
                                scores.append(scorer_data['score'])
                            if 'success' in scorer_data:
                                successes.append(scorer_data['success'])
                
                if scores:
                    analysis["evaluation_scores"][scorer_name] = {
                        "average_score": sum(scores) / len(scores),
                        "min_score": min(scores),
                        "max_score": max(scores),
                        "success_rate": sum(successes) / len(successes) if successes else 0,
                        "total_evaluated": len(scores)
                    }
        
        # NEW: Analyze trace capture and execution patterns
        trace_captured_count = 0
        total_trace_spans = 0
        execution_patterns = {}
        
        for result in results:
            # Check if result has evaluation scores with metadata
            if result.evaluation_scores:
                for scorer_name, scorer_data in result.evaluation_scores.items():
                    if isinstance(scorer_data, dict):
                        metadata = scorer_data.get('additional_metadata', {})
                        if isinstance(metadata, dict):
                            if metadata.get('trace_captured', False):
                                trace_captured_count += 1
                            total_trace_spans += metadata.get('trace_spans_count', 0)
                            
                            # Analyze execution patterns
                            exec_seq = metadata.get('execution_sequence', [])
                            if exec_seq:
                                pattern = ' -> '.join([item.get('tool_name', 'unknown') for item in exec_seq])
                                execution_patterns[pattern] = execution_patterns.get(pattern, 0) + 1
        
        analysis["trace_analysis"] = {
            "trace_capture_rate": (trace_captured_count / len(results)) * 100 if results else 0,
            "total_trace_spans": total_trace_spans,
            "average_spans_per_query": total_trace_spans / len(results) if results else 0,
            "execution_patterns": execution_patterns,
            "trace_enabled": trace_captured_count > 0
        }
        
        # Identify issues
        for result in results:
            if not result.success:
                analysis["issues_found"].append({
                    "query": result.query,
                    "errors": result.errors,
                    "type": "system_failure"
                })
            
            # Check for low confidence responses
            if result.confidence < 0.5:
                analysis["issues_found"].append({
                    "query": result.query,
                    "confidence": result.confidence,
                    "type": "low_confidence"
                })
            
            # Check for evaluation failures
            if result.evaluation_scores:
                for scorer_name, scorer_data in result.evaluation_scores.items():
                    if isinstance(scorer_data, dict) and not scorer_data.get('success', True):
                        analysis["issues_found"].append({
                            "query": result.query,
                            "scorer": scorer_name,
                            "reason": scorer_data.get('reason', 'Unknown'),
                            "type": "evaluation_failure"
                        })
        
        return analysis
    
    def print_evaluation_report(self, results: List[TennisEvaluationResult]):
        """Print a comprehensive evaluation report."""
        analysis = self.analyze_evaluation_results(results)
        
        print("\n" + "="*80)
        print("🎾 TENNIS INTELLIGENCE SYSTEM - EVALUATION REPORT")
        print("="*80)
        
        # Summary
        summary = analysis["summary"]
        print(f"\n📊 SUMMARY:")
        print(f"   • Total Queries: {summary['total_queries']}")
        print(f"   • Successful: {summary['successful_queries']} ({summary['successful_queries']/summary['total_queries']*100:.1f}%)")
        print(f"   • Failed: {summary['failed_queries']} ({summary['failed_queries']/summary['total_queries']*100:.1f}%)")
        print(f"   • Average Confidence: {summary['average_confidence']:.2f}")
        print(f"   • Average Processing Time: {summary['average_processing_time']:.2f}s")
        
        # Tool Usage
        if analysis["tool_usage"]:
            print(f"\n🔧 TOOL USAGE:")
            for tool, stats in analysis["tool_usage"].items():
                print(f"   • {tool}: {stats['count']} times ({stats['percentage']:.1f}%)")
        
        # Evaluation Scores
        if analysis["evaluation_scores"]:
            print(f"\n🔍 EVALUATION SCORES:")
            for scorer, stats in analysis["evaluation_scores"].items():
                print(f"   • {scorer}:")
                print(f"     - Average Score: {stats['average_score']:.3f}")
                print(f"     - Success Rate: {stats['success_rate']:.1%}")
                print(f"     - Range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
        
        # NEW: Trace Analysis
        if analysis["trace_analysis"]["trace_enabled"]:
            print(f"\n📊 TRACE ANALYSIS:")
            trace_stats = analysis["trace_analysis"]
            print(f"   • Trace Capture Rate: {trace_stats['trace_capture_rate']:.1f}%")
            print(f"   • Total Trace Spans: {trace_stats['total_trace_spans']}")
            print(f"   • Avg Spans per Query: {trace_stats['average_spans_per_query']:.1f}")
            if trace_stats['execution_patterns']:
                print(f"   • Execution Patterns:")
                for pattern, count in trace_stats['execution_patterns'].items():
                    print(f"     - {pattern}: {count} times")
        
        # Issues Found
        if analysis["issues_found"]:
            print(f"\n⚠️  ISSUES FOUND ({len(analysis['issues_found'])}):")
            issue_types = {}
            for issue in analysis["issues_found"]:
                issue_type = issue["type"]
                if issue_type not in issue_types:
                    issue_types[issue_type] = 0
                issue_types[issue_type] += 1
            
            for issue_type, count in issue_types.items():
                print(f"   • {issue_type}: {count} issues")
        
        print("\n" + "="*80)
    
    def save_evaluation_results(self, 
                              results: List[TennisEvaluationResult], 
                              filepath: Optional[str] = None):
        """Save evaluation results to JSON file."""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"tennis_evaluation_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                "query": result.query,
                "response": result.response,
                "confidence": result.confidence,
                "sources": result.sources,
                "tools_called": result.tools_called,
                "processing_time": result.processing_time,
                "evaluation_scores": result.evaluation_scores,
                "success": result.success,
                "errors": result.errors
            })
        
        # Add analysis
        analysis = self.analyze_evaluation_results(results)
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "project_name": self.project_name,
            "total_queries": len(results),
            "analysis": analysis,
            "detailed_results": serializable_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Evaluation results saved to: {filepath}")


def main():
    """Main function to run tennis system evaluation."""
    print("🎾 Tennis Intelligence System - JudgeVal Evaluation")
    print("=" * 60)
    print("✨ Features: LangGraph trace capture and tennis expertise evaluation!")
    print("   • Trace capture for execution flow analysis")
    print("   • Tennis expertise classification")  
    print("   • Faithfulness and relevancy scoring")
    print("   • Tool usage and dependency validation")
    print("💡 For advanced scorers, upgrade: pip install --upgrade judgeval")
    print("=" * 60)
    
    # Check for required environment variables
    if not os.getenv('JUDGMENT_API_KEY'):
        print("❌ JUDGMENT_API_KEY environment variable not set")
        print("Please set your JudgeVal API key:")
        print("export JUDGMENT_API_KEY='your-api-key'")
        return
    
    if not os.getenv('JUDGMENT_ORG_ID'):
        print("❌ JUDGMENT_ORG_ID environment variable not set")
        print("Please set your JudgeVal organization ID:")
        print("export JUDGMENT_ORG_ID='your-org-id'")
        return
    
    # Initialize evaluation suite
    debug_mode = os.environ.get('TENNIS_DEBUG', 'False').lower() == 'true'
    eval_suite = TennisEvaluationSuite(debug=debug_mode)
    
    try:
        # Run evaluation
        print("\n🚀 Starting comprehensive evaluation...")
        results = eval_suite.run_tennis_evaluation()
        
        # Print report
        eval_suite.print_evaluation_report(results)
        
        # Save results
        eval_suite.save_evaluation_results(results)
        
        print("\n✅ Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
