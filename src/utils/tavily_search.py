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
    client = TavilyClient(api_key="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") # Ideally I would ask user to input their own api key, but leaving this one here to facilitate more plug and play demo without additional setups. 
    results = client.search(query)
    _debug_print("\nTavily search results:\n")
    _debug_print(str(results))
    _debug_print("\n--------------------------------\n")

    return extract_tavily_search_results(results)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        input_query = " ".join(sys.argv[1:])
    else:
        input_query = "Who is the best tennis player in the world right now?"

    tavily_search(query=input_query)