"""
Tennis Intelligence System - Judgeval Evaluation Implementation
=============================================================

This script implements comprehensive evaluation of the tennis intelligence system
using judgeval scorers to assess answer quality, relevancy, faithfulness, and correctness.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Add src directory to Python path for proper imports
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Change directory to project root for relative file paths
os.chdir(current_dir)

# Import judgeval components
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    FaithfulnessScorer,
    AnswerRelevancyScorer, 
    AnswerCorrectnessScorer,
    HallucinationScorer,
    InstructionAdherenceScorer,
    GroundednessScorer,
    ToolOrderScorer,
    ToolDependencyScorer
)

# Import tennis system components
try:
    from tennis_agents import TennisIntelligenceSystem
    from utils.simple_memory_manager import create_session_id
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure you're running from the eval_agent directory")
    sys.exit(1)


class TennisAgentsEvaluator:
    """
    Comprehensive evaluator for the Tennis Intelligence System using judgeval.
    
    Tests various aspects of the system:
    - Answer relevancy to tennis queries
    - Faithfulness to retrieved data
    - Answer correctness
    - Hallucination detection
    - Instruction adherence
    - Groundedness in sources
    """
    
    def __init__(self):
        """Initialize the evaluator with tennis system and judgeval client."""
        print("🎾 Initializing Tennis Agents Evaluator...")
        
        # Initialize tennis intelligence system with LangGraph for proper tool calling
        self.tennis_system = TennisIntelligenceSystem()
        
        # Initialize judgeval client
        self.judgment_client = JudgmentClient()
        
        # Initialize scorers with appropriate thresholds
        self.scorers = [
            AnswerRelevancyScorer(threshold=0.7),      # Is answer relevant to tennis query?
            FaithfulnessScorer(threshold=0.8),         # Is answer faithful to retrieved data?
            AnswerCorrectnessScorer(threshold=0.7),    # Is the answer factually correct?
            HallucinationScorer(threshold=0.3),        # Low threshold - we want to catch hallucinations
            InstructionAdherenceScorer(threshold=0.8), # Does it follow tennis-specific instructions?
            GroundednessScorer(threshold=0.7),          # Is answer grounded in provided sources?
            ToolOrderScorer(threshold=0.8),           # Are tools called in the correct order?
            ToolDependencyScorer(threshold=0.8)       # Are tools dependent on each other?
        ]
        
        print("✅ Tennis Agents Evaluator initialized successfully")
        print(f"📊 Configured {len(self.scorers)} evaluation scorers")
        print("🚀 Using LangGraph system with official tool calling")
    
    def create_test_examples(self) -> List[Example]:
        """
        Create comprehensive test examples covering different tennis query types.
        
        Returns:
            List of Example objects for evaluation
        """
        test_queries = [
            {
                "input": "Who won the most Grand Slam titles in men's tennis?",
                "description": "Statistical query about historical tennis records",
                "expected_topics": ["Novak Djokovic", "Rafael Nadal", "Roger Federer", "Grand Slam"],
                "context_type": "historical_stats"
            },
            {
                "input": "What is Novak Djokovic's head-to-head record against Rafael Nadal?", 
                "description": "Head-to-head statistical query",
                "expected_topics": ["Djokovic", "Nadal", "head-to-head", "wins", "losses"],
                "context_type": "player_comparison"
            },
            {
                "input": "Who is the current world number 1 in men's tennis?",
                "description": "Current ranking query requiring recent information",
                "expected_topics": ["ranking", "ATP", "number 1", "current"],
                "context_type": "current_rankings"
            },
            {
                "input": "What surface does Rafael Nadal perform best on?",
                "description": "Surface performance analysis",
                "expected_topics": ["Nadal", "clay", "surface", "French Open", "performance"],
                "context_type": "surface_analysis"
            },
            {
                "input": "Which tennis players have won all four Grand Slams?",
                "description": "Achievement-based query about Career Grand Slam",
                "expected_topics": ["Career Grand Slam", "Wimbledon", "US Open", "French Open", "Australian Open"],
                "context_type": "achievement_query"
            },
            {
                "input": "What are the major tennis tournaments?",
                "description": "General tennis knowledge query",
                "expected_topics": ["Grand Slam", "ATP Masters", "tournaments", "Wimbledon", "US Open"],
                "context_type": "general_knowledge"
            },
            {
                "input": "How many sets are played in a men's Grand Slam match?",
                "description": "Tennis rules and format query",
                "expected_topics": ["best of five", "sets", "Grand Slam", "men's"],
                "context_type": "rules_format"
            },
            {
                "input": "Who has the fastest serve in tennis history?",
                "description": "Tennis record query about serve speed",
                "expected_topics": ["serve speed", "fastest", "mph", "km/h", "record"],
                "context_type": "performance_records"
            }
        ]
        
        examples = []
        session_id = create_session_id()
        
        print("🔄 Generating tennis system responses for evaluation...")
        
        for i, test_case in enumerate(test_queries):
            print(f"Processing query {i+1}/{len(test_queries)}: {test_case['input']}")
            
            # Get response from tennis system
            start_time = time.time()
            result = self.tennis_system.process_query(test_case["input"], session_id)
            processing_time = time.time() - start_time
            
            # Extract key information from the result
            actual_output = result.get('response', '')
            sources = result.get('sources', [])
            confidence = result.get('confidence', 0.0)
            sql_data_used = result.get('sql_data_used', False)
            search_data_used = result.get('search_data_used', False)
            tools_called_actual = result.get('tools_called', [])  # From LangGraph system
            langgraph_used = result.get('langgraph_used', False)
            
            # Create tools_called list based on what the system actually used
            tools_called = tools_called_actual if tools_called_actual else []
            expected_tools = []
            
            # Determine expected tools based on query type
            query_type = test_case["context_type"]
            if query_type in ["historical_stats", "player_comparison", "surface_analysis", "achievement_query"]:
                expected_tools.extend(["generate_sql_query", "execute_sql_query", "interpret_sql_results"])
            elif query_type in ["current_rankings", "general_knowledge", "rules_format", "performance_records"]:
                expected_tools.extend(["optimize_search_query", "tavily_search_tool", "interpret_search_results"])
            else:
                # Mixed or unknown - could use either
                expected_tools.extend(["generate_sql_query", "tavily_search_tool"])
            
            # Create retrieval context from sources and metadata
            retrieval_context = []
            
            # Add source information
            if sources:
                retrieval_context.append(f"Information sources: {', '.join(sources)}")
            
            # Add confidence information
            retrieval_context.append(f"System confidence: {confidence:.2f}")
            
            # Add processing metadata
            if sql_data_used:
                retrieval_context.append("Used tennis database for historical/statistical data")
            if search_data_used:
                retrieval_context.append("Used web search for current information")
            
            # Add LangGraph information
            if langgraph_used:
                retrieval_context.append("Processed using LangGraph workflow with official tool calling")
            
            # Add expected topics for evaluation context
            retrieval_context.append(f"Expected topics: {', '.join(test_case['expected_topics'])}")
            retrieval_context.append(f"Query type: {test_case['context_type']}")
            
            # Add tool usage information to context for hallucination scorer
            if tools_called:
                retrieval_context.append(f"Tools executed: {', '.join(tools_called)}")
            
            # Create context field for hallucination scorer (required parameter)
            context_for_hallucination = retrieval_context.copy()
            
            # Create expected output based on query type and topics
            expected_output = self._generate_expected_output(test_case)
            
            # Create Example for judgeval
            example = Example(
                input=test_case["input"],
                actual_output=actual_output,
                expected_output=expected_output,
                context=context_for_hallucination,  # Required for HallucinationScorer
                retrieval_context=retrieval_context,
                tools_called=tools_called if tools_called else None,
                expected_tools=expected_tools if expected_tools else None,
                additional_metadata={
                    "description": test_case["description"],
                    "context_type": test_case["context_type"],
                    "processing_time": processing_time,
                    "system_confidence": confidence,
                    "sources_used": sources,
                    "sql_data_used": sql_data_used,
                    "search_data_used": search_data_used,
                    "tools_actually_called": tools_called,
                    "tools_expected": expected_tools,
                    "langgraph_used": langgraph_used
                }
            )
            
            examples.append(example)
            print(f"  ✅ Generated example with {len(retrieval_context)} context items")
        
        print(f"🎯 Created {len(examples)} test examples for evaluation")
        return examples
    
    def _generate_expected_output(self, test_case: Dict[str, Any]) -> str:
        """
        Generate expected output for comparison based on query type.
        
        Args:
            test_case: Test case dictionary with input and metadata
            
        Returns:
            Expected output string for evaluation
        """
        query_type = test_case["context_type"]
        topics = test_case["expected_topics"]
        
        if query_type == "historical_stats":
            return f"A factual answer about tennis statistics mentioning key players like {', '.join(topics[:3])} and providing specific numbers or achievements."
        
        elif query_type == "player_comparison":
            return f"A head-to-head comparison providing specific win-loss records between the mentioned players with factual data."
        
        elif query_type == "current_rankings":
            return f"Current ranking information identifying the world number 1 player with recent data from official tennis rankings."
        
        elif query_type == "surface_analysis":
            return f"Surface-specific performance analysis mentioning the player's best surface with supporting statistics or achievements."
        
        elif query_type == "achievement_query":
            return f"A list of players who achieved the Career Grand Slam with factual information about their accomplishments."
        
        elif query_type == "general_knowledge":
            return f"Comprehensive information about major tennis tournaments including the four Grand Slams and other important events."
        
        elif query_type == "rules_format":
            return f"Clear explanation of tennis match format rules providing accurate information about sets and scoring."
        
        elif query_type == "performance_records":
            return f"Information about tennis performance records with specific data about serve speeds and record holders."
        
        else:
            return f"A comprehensive tennis-related answer covering the expected topics: {', '.join(topics)}"
    
    def run_evaluation(self, project_name: str = "tennis_agents_eval") -> Dict[str, Any]:
        """
        Run comprehensive evaluation of the tennis intelligence system.
        
        Args:
            project_name: Name for the evaluation project
            
        Returns:
            Dictionary with evaluation results and analysis
        """
        print(f"\n🚀 Starting Tennis Agents Evaluation - Project: {project_name}")
        print("=" * 60)
        
        # Create test examples
        examples = self.create_test_examples()
        
        # Run judgeval evaluation
        print("\n📊 Running judgeval evaluation...")
        eval_start_time = time.time()
        
        try:
            results = self.judgment_client.run_evaluation(
                examples=examples,
                scorers=self.scorers,
                model="gpt-4o",  # Use high-quality model for evaluation
                project_name=project_name,
                eval_run_name=f"tennis_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                override=True
            )
            
            eval_duration = time.time() - eval_start_time
            print(f"✅ Evaluation completed in {eval_duration:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {"error": str(e)}
        
        # Analyze and summarize results
        analysis = self._analyze_results(results, examples)
        
        # Display results
        self._display_results(analysis)
        
        return analysis
    
    def _analyze_results(self, results: List, examples: List[Example]) -> Dict[str, Any]:
        """
        Analyze evaluation results and create comprehensive summary.
        
        Args:
            results: List of ScoringResult objects from judgeval
            examples: List of Example objects that were evaluated
            
        Returns:
            Dictionary with detailed analysis
        """
        print("\n🔍 Analyzing evaluation results...")
        
        analysis = {
            "overall_metrics": {},
            "scorer_breakdown": {},
            "query_type_performance": {},
            "individual_results": [],
            "recommendations": []
        }
        
        # Initialize counters
        total_examples = len(examples)
        scorer_totals = {scorer.__class__.__name__.replace("Scorer", ""): {"passed": 0, "total": 0, "scores": []} 
                        for scorer in self.scorers}
        
        query_type_performance = {}
        
        # Analyze each result
        for i, (result, example) in enumerate(zip(results, examples)):
            query_type = example.additional_metadata.get("context_type", "unknown")
            
            if query_type not in query_type_performance:
                query_type_performance[query_type] = {"passed": 0, "total": 0, "avg_score": 0, "scores": []}
            
            example_analysis = {
                "query": example.input,
                "query_type": query_type,
                "system_response": example.actual_output,
                "system_confidence": example.additional_metadata.get("system_confidence", 0),
                "sources_used": example.additional_metadata.get("sources_used", []),
                "scorer_results": {}
            }
            
            # Analyze individual scorer results
            for scorer_data in result.scorer_data:
                scorer_name = scorer_data.score_type.replace("_", " ").title().replace(" ", "")
                if scorer_name.endswith("Scorer"):
                    scorer_name = scorer_name[:-6]  # Remove "Scorer" suffix
                
                score = scorer_data.score
                success = scorer_data.success
                reason = scorer_data.reason
                
                # Update scorer totals
                if scorer_name in scorer_totals:
                    scorer_totals[scorer_name]["total"] += 1
                    scorer_totals[scorer_name]["scores"].append(score)
                    if success:
                        scorer_totals[scorer_name]["passed"] += 1
                
                # Update query type performance
                query_type_performance[query_type]["total"] += 1
                query_type_performance[query_type]["scores"].append(score)
                if success:
                    query_type_performance[query_type]["passed"] += 1
                
                example_analysis["scorer_results"][scorer_name] = {
                    "score": score,
                    "passed": success,
                    "reason": reason
                }
            
            analysis["individual_results"].append(example_analysis)
        
        # Calculate overall metrics
        total_tests = sum(data["total"] for data in scorer_totals.values())
        total_passed = sum(data["passed"] for data in scorer_totals.values())
        
        analysis["overall_metrics"] = {
            "total_examples": total_examples,
            "total_tests": total_tests,
            "overall_pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "average_system_confidence": sum(ex.additional_metadata.get("system_confidence", 0) 
                                           for ex in examples) / len(examples)
        }
        
        # Calculate scorer breakdown
        for scorer_name, data in scorer_totals.items():
            if data["total"] > 0:
                analysis["scorer_breakdown"][scorer_name] = {
                    "pass_rate": (data["passed"] / data["total"] * 100),
                    "average_score": sum(data["scores"]) / len(data["scores"]),
                    "total_tests": data["total"],
                    "passed_tests": data["passed"]
                }
        
        # Calculate query type performance
        for query_type, data in query_type_performance.items():
            if data["total"] > 0:
                analysis["query_type_performance"][query_type] = {
                    "pass_rate": (data["passed"] / data["total"] * 100),
                    "average_score": sum(data["scores"]) / len(data["scores"]),
                    "total_tests": data["total"]
                }
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on evaluation results.
        
        Args:
            analysis: Analysis dictionary with results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        scorer_breakdown = analysis.get("scorer_breakdown", {})
        query_performance = analysis.get("query_type_performance", {})
        
        # Check scorer performance
        for scorer_name, data in scorer_breakdown.items():
            pass_rate = data["pass_rate"]
            avg_score = data["average_score"]
            
            if pass_rate < 70:
                if scorer_name == "Hallucination":
                    recommendations.append(f"🚨 High hallucination detected ({pass_rate:.1f}% pass rate). Review data grounding and factual accuracy.")
                elif scorer_name == "Faithfulness":
                    recommendations.append(f"📚 Low faithfulness score ({pass_rate:.1f}% pass rate). Improve adherence to source data.")
                elif scorer_name == "AnswerRelevancy":
                    recommendations.append(f"🎯 Low relevancy score ({pass_rate:.1f}% pass rate). Enhance query understanding and routing.")
                elif scorer_name == "AnswerCorrectness":
                    recommendations.append(f"✅ Low correctness score ({pass_rate:.1f}% pass rate). Verify factual accuracy of responses.")
                elif scorer_name == "ToolOrder":
                    recommendations.append(f"⚠️ Tool ordering issue detected ({pass_rate:.1f}% pass rate). Ensure tools are called in the correct sequence.")
                elif scorer_name == "ToolDependency":
                    recommendations.append(f"⚠️ Tool dependency issue detected ({pass_rate:.1f}% pass rate). Ensure tools are called in the correct order.")
                else:
                    recommendations.append(f"⚠️ {scorer_name} needs improvement ({pass_rate:.1f}% pass rate).")
        
        # Check query type performance
        for query_type, data in query_performance.items():
            pass_rate = data["pass_rate"]
            if pass_rate < 70:
                recommendations.append(f"🔧 {query_type.replace('_', ' ').title()} queries need improvement ({pass_rate:.1f}% pass rate).")
        
        # Overall performance recommendations
        overall_pass_rate = analysis["overall_metrics"]["overall_pass_rate"]
        if overall_pass_rate > 85:
            recommendations.append("🎉 Excellent overall performance! Consider fine-tuning for edge cases.")
        elif overall_pass_rate > 70:
            recommendations.append("✅ Good overall performance. Focus on specific weak areas identified above.")
        else:
            recommendations.append("🔧 Overall performance needs improvement. Review system architecture and data sources.")
        
        return recommendations
    
    def _analyze_tool_usage(self, individual_results: List[Dict[str, Any]]) -> List[str]:
        """
        Analyze tool usage across all test examples.
        
        Args:
            individual_results: List of individual result dictionaries
            
        Returns:
            List of tool usage summary strings
        """
        tool_stats = {}
        total_examples = len(individual_results)
        
        for result in individual_results:
            metadata = result.get("additional_metadata", {})
            tools_called = metadata.get("tools_actually_called", [])
            
            for tool in tools_called:
                if tool not in tool_stats:
                    tool_stats[tool] = 0
                tool_stats[tool] += 1
        
        summary = []
        
        if tool_stats:
            summary.append(f"📊 Tool Usage Summary ({total_examples} examples):")
            for tool, count in sorted(tool_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_examples) * 100
                summary.append(f"   • {tool}: {count}/{total_examples} ({percentage:.1f}%)")
        else:
            summary.append("⚠️ No tool usage detected - check tool calling implementation")
        
        # Analyze tool calling patterns
        sql_usage = sum(1 for result in individual_results 
                       if any('sql' in tool.lower() for tool in result.get("additional_metadata", {}).get("tools_actually_called", [])))
        search_usage = sum(1 for result in individual_results 
                          if any('search' in tool.lower() for tool in result.get("additional_metadata", {}).get("tools_actually_called", [])))
        
        summary.append(f"🔍 Pattern Analysis:")
        summary.append(f"   • SQL Tools Used: {sql_usage}/{total_examples} examples")
        summary.append(f"   • Search Tools Used: {search_usage}/{total_examples} examples")
        
        return summary
    
    def _display_results(self, analysis: Dict[str, Any]) -> None:
        """
        Display evaluation results in a formatted manner.
        
        Args:
            analysis: Analysis dictionary with results
        """
        print("\n" + "=" * 60)
        print("🎾 TENNIS AGENTS EVALUATION RESULTS")
        print("=" * 60)
        
        # Overall metrics
        overall = analysis["overall_metrics"]
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Examples: {overall['total_examples']}")
        print(f"   Total Tests: {overall['total_tests']}")
        print(f"   Overall Pass Rate: {overall['overall_pass_rate']:.1f}%")
        print(f"   Avg System Confidence: {overall['average_system_confidence']:.2f}")
        
        # Scorer breakdown
        print(f"\n🔍 SCORER BREAKDOWN:")
        for scorer_name, data in analysis["scorer_breakdown"].items():
            status_icon = "✅" if data["pass_rate"] >= 70 else "⚠️" if data["pass_rate"] >= 50 else "❌"
            print(f"   {status_icon} {scorer_name}: {data['pass_rate']:.1f}% pass rate (avg score: {data['average_score']:.2f})")
        
        # Query type performance
        print(f"\n🎯 QUERY TYPE PERFORMANCE:")
        for query_type, data in analysis["query_type_performance"].items():
            status_icon = "✅" if data["pass_rate"] >= 70 else "⚠️" if data["pass_rate"] >= 50 else "❌"
            display_name = query_type.replace('_', ' ').title()
            print(f"   {status_icon} {display_name}: {data['pass_rate']:.1f}% pass rate (avg score: {data['average_score']:.2f})")
        
        # Recommendations
        if analysis["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in analysis["recommendations"]:
                print(f"   {rec}")
        
        # Individual results summary
        print(f"\n📋 INDIVIDUAL RESULTS SUMMARY:")
        for i, result in enumerate(analysis["individual_results"][:5]):  # Show first 5
            print(f"\n   Query {i+1}: {result['query'][:60]}...")
            print(f"   Type: {result['query_type'].replace('_', ' ').title()}")
            print(f"   System Confidence: {result['system_confidence']:.2f}")
            
            # Show tool usage if available
            tools_used = result.get("tools_used", [])
            if tools_used:
                print(f"   Tools Used: {', '.join(tools_used)}")
            
            # Show worst performing scorer for this example
            worst_scorer = min(result["scorer_results"].items(), 
                             key=lambda x: x[1]["score"])
            best_scorer = max(result["scorer_results"].items(), 
                            key=lambda x: x[1]["score"])
            
            print(f"   Best Score: {best_scorer[0]} ({best_scorer[1]['score']:.2f})")
            print(f"   Worst Score: {worst_scorer[0]} ({worst_scorer[1]['score']:.2f})")
        
        if len(analysis["individual_results"]) > 5:
            print(f"   ... and {len(analysis['individual_results']) - 5} more results")
        
        # Add tool calling analysis
        print(f"\n🔧 TOOL CALLING ANALYSIS:")
        tool_usage_summary = self._analyze_tool_usage(analysis["individual_results"])
        for tool_stat in tool_usage_summary:
            print(f"   {tool_stat}")
        
        print("\n" + "=" * 60)


def main():
    """Main function to run the tennis agents evaluation."""
    print("🎾 Tennis Intelligence System - Judgeval Evaluation")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = TennisAgentsEvaluator()
        
        # Run evaluation
        project_name = f"tennis_agents_eval_{datetime.now().strftime('%Y%m%d')}"
        results = evaluator.run_evaluation(project_name)
        
        if "error" not in results:
            print("\n🎉 Evaluation completed successfully!")
            print(f"📊 Results saved to judgeval project: {project_name}")
            
            # Save detailed results to file
            results_file = f"tennis_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Detailed results saved to: {results_file}")
        else:
            print(f"❌ Evaluation failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
