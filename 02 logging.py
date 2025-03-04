import logging
from langchain_ollama import OllamaLLM
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from googlesearch import search

# إعداد نظام تسجيل الأخطاء
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize LLM before using it in `google_search`
llm = OllamaLLM(model="llama3", temperature=0)  # model: LLM model to use, temperature: Controls randomness

def google_search(query: str, num_results: int = 2):
    """Search Google and summarize results using LLM."""
    logger.info(f"Received search query: {query}")

    try:
        results = list(search(query, stop=num_results))  # num_results: Number of search results to retrieve
        if not results:
            logger.warning("No results found for query.")
            return "❌ No results found."

        # Passing results to LLM for summarization
        summary_prompt = (
            f"Summarize the key information about '{query}' based on these sources: {results}"
        )
        summary = llm.invoke(summary_prompt)

        logger.debug(f"Search summary: {summary}")
        return summary

    except Exception as e:
        logger.error(f"Search failed due to: {e}", exc_info=True)
        return "❌ An error occurred while searching."

# Define toolset
def search_tool_handler(query):
    """Handles input for google_search tool, ensuring correct format."""
    if isinstance(query, str):
        return google_search(query)
    elif isinstance(query, dict) and "value" in query:
        return google_search(query["value"])
    else:
        logger.warning("Invalid input format for search tool.")
        return "❌ Invalid input format for search tool."

# تحديث تعريف الأداة باستخدام الدالة الجديدة
tools = [
    Tool(
        name="google_search",
        func=search_tool_handler,
        description="Search Google using a query and return summarized results."
    )
]

# Initialize agent
agent = initialize_agent(
    tools=tools,  # tools: List of available tools
    llm=llm,  # llm: Language model used for decision making
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # agent: Type of agent behavior
    verbose=True,  # verbose: Enables detailed logging
    handle_parsing_errors=True,  # handle_parsing_errors: Manages parsing errors automatically
    max_iterations=2,  # max_iterations: Maximum loops for decision-making
    max_execution_time=None  # max_execution_time: No time limit for execution
)

if __name__ == "__main__":
    while True:
        query = input("\n🔎 Enter your search query (or type 'exit' to exit): ").strip()
        if query.lower() == "exit":
            print("👋 The program has been terminated.")
            break

        try:
            response = agent.invoke({"input": query, "chat_history": []})  # input: Search query, chat_history: Conversation history

            print("\n🔍 Search results from the agent:")
            print(response)
        except Exception as e:
            logger.error(f"❌ Error during execution: {e}", exc_info=True)
            print("❌ An error occurred during execution.")
