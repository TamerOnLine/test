import logging
from langchain_ollama import OllamaLLM
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from googlesearch import search

# إعداد نظام تسجيل الأخطاء
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# محاول تهيئة LLM مع تسجيل الأخطاء
try:
    llm = OllamaLLM(model="llama3", temperature=0)
    logger.info("✅ LLM initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM: {e}", exc_info=True)
    llm = None  # تجنب انهيار البرنامج في حالة الفشل

def google_search(query: str, num_results: int = 2):
    """Search Google and summarize results using LLM."""
    logger.info(f"Received search query: {query}")

    try:
        results = list(search(query, stop=num_results))
        if not results:
            logger.warning("No results found for query.")
            return "❌ No results found."

        if llm:
            summary_prompt = f"Summarize the key information about '{query}' based on these sources: {results}"
            summary = llm.invoke(summary_prompt)
            logger.debug(f"Search summary: {summary}")
            return summary
        else:
            logger.warning("LLM is not initialized. Returning raw results.")
            return results

    except Exception as e:
        logger.error(f"Search failed due to: {e}", exc_info=True)
        return "❌ An error occurred while searching."

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

# Initialize agent فقط إذا كان LLM مهيأ بشكل صحيح
if llm:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=2,
        max_execution_time=None
    )
else:
    logger.error("❌ Agent initialization failed due to LLM issue.")

if __name__ == "__main__":
    print("\n🔎 Welcome to the AI Search Agent! Type 'exit' to quit.\n")
    
    try:
        while True:
            query = input("\n🔎 Enter your search query (or type 'exit' to exit): ").strip()
            if query.lower() == "exit":
                print("\n👋 Exiting the program. Have a great day!")
                break

            if not query:
                print("⚠️ Please enter a valid query.")
                continue

            try:
                if llm:
                    response = agent.invoke({"input": query, "chat_history": []})
                    print("\n🔍 Search results from the agent:\n---------------------")
                    print(response)
                    print("---------------------\n")
                else:
                    print("❌ LLM is not available. Please check your setup.")
            except Exception as e:
                logger.error(f"❌ Error during execution: {e}", exc_info=True)
                print("❌ An error occurred during execution.")
    except KeyboardInterrupt:
        print("\n\n👋 Exiting the program. See you next time!")

