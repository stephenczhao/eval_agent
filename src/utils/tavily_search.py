from tavily import TavilyClient
import sys
import os

def _debug_print(message: str) -> None:
    """Print debug messages only when debug mode is enabled"""
    if os.getenv('TENNIS_DEBUG', 'false').lower() == 'true':
        print(message)

def extract_tavily_search_results(results: dict):
    # todo: add a langraph agent to extract and summarize the returned tavily search results into effective sentences. 
    return results

def tavily_search(query: str):
    """
    Perform web search using Tavily API with improved error handling and query optimization.
    
    Args:
        query: Search query string
        
    Returns:
        Search results dictionary
    """
    try:
        # Clean up the query - remove any site restrictions that might be too limiting
        cleaned_query = query.replace('site:atptour.com OR site:wtatennis.com OR site:espn.com', '')
        cleaned_query = cleaned_query.replace('"', '').strip()
        
        # If query is too restrictive or specific, make it more general
        if not cleaned_query or len(cleaned_query.split()) < 2:
            cleaned_query = f"tennis {query}"
            
        _debug_print(f"🔍 Original query: {query}")
        _debug_print(f"🔍 Cleaned query: {cleaned_query}")
        
        client = TavilyClient(api_key="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")
        
        # Try the cleaned query first
        results = client.search(cleaned_query, max_results=5)
        _debug_print("\nTavily search results:\n")
        _debug_print(str(results))
        _debug_print("\n--------------------------------\n")
        
        # If no results, try a simpler fallback query
        if not results.get("results") or len(results.get("results", [])) == 0:
            _debug_print("🔄 No results found, trying fallback query...")
            fallback_query = "tennis rankings latest news"
            results = client.search(fallback_query, max_results=5)
            _debug_print(f"🔄 Fallback query: {fallback_query}")
            _debug_print(f"🔄 Fallback results: {len(results.get('results', []))} items")
        
        return extract_tavily_search_results(results)
        
    except Exception as e:
        _debug_print(f"❌ Tavily search error: {str(e)}")
        # Return empty results structure instead of crashing
        return {
            "results": [],
            "query": query,
            "response_time": 0,
            "error": str(e)
        }

if __name__ == "__main__":

    if len(sys.argv) > 1:
        input_query = " ".join(sys.argv[1:])
    else:
        input_query = "Who is the best tennis player in the world right now?"

    tavily_search(query=input_query)