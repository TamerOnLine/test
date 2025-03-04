import logging
from langchain_ollama import OllamaLLM
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from googlesearch import search
import warnings
import textwrap
import time

# إعداد نظام تسجيل الأخطاء
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



# إخفاء تحذيرات LangChain مع تسجيلها في الـ logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
logger.warning("⚠️ LangChainDeprecationWarning: It's recommended to migrate to LangGraph for new use cases.")


# محاول تهيئة LLM مع تسجيل الأخطاء
try:
    llm = OllamaLLM(model="llama3", temperature=0)
    logger.info("✅ LLM initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM: {e}", exc_info=True)
    llm = None  # تجنب انهيار البرنامج في حالة الفشل

def google_search(query: str, num_results: int = 2, use_llm: bool = True):
    """Search Google and summarize results using LLM (if enabled)."""
    logger.info(f"Received search query: {query}")

    try:
        results = list(search(query, stop=num_results))
        if not results:
            logger.warning("No results found for query.")
            return "❌ No results found."

        # إذا لم يكن LLM مفعّلًا، نرجع النتائج فقط
        if not use_llm or not llm:
            return f"🔍 Top search results:\n{results}"

        # تمرير النتائج إلى LLM فقط إذا تم تمكينه
        summary_prompt = f"Summarize the key information about '{query}' based on these sources: {results}"
        summary = llm.invoke(summary_prompt)

        logger.debug(f"Search summary: {summary}")
        return summary

    except Exception as e:
        logger.error(f"Search failed due to: {e}", exc_info=True)
        return "❌ An error occurred while searching."

def search_tool_handler(query):
    """Handles input for google_search tool, ensuring correct format."""
    use_llm = True  # افتراضيًا يتم تفعيل LLM
    
    if isinstance(query, dict):
        # السماح للمستخدم بتعطيل LLM إذا أراد
        use_llm = query.get("use_llm", True)
        query_value = query.get("value", "")
    elif isinstance(query, str):
        query_value = query
    else:
        logger.warning("Invalid input format for search tool.")
        return "❌ Invalid input format for search tool."

    return google_search(query_value, use_llm=use_llm)

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



def print_search_results(response):
    """Formats and prints search results in a clean way."""
    input_text = response.get("input", "Unknown Query")
    output_text = response.get("output", "No response available.")

    print("\n🔍 Search Query:")
    print(f"   {input_text}")

    print("\n📌 AI Response:")
    print(textwrap.fill(output_text, width=80))  # تحديد عرض النص لتحسين القراءة

    print("\n---------------------\n")




# متغير لحساب الوقت بين كل استعلام
last_search_time = 0
SEARCH_DELAY = 3  # تأخير 3 ثوانٍ بين كل بحث وآخر

def enforce_rate_limit():
    """Ensures that searches are not too frequent."""
    global last_search_time
    elapsed_time = time.perf_counter() - last_search_time
    if elapsed_time < SEARCH_DELAY:
        wait_time = SEARCH_DELAY - elapsed_time
        print(f"⏳ Waiting {wait_time:.1f} seconds before next search...")
        time.sleep(wait_time)
    last_search_time = time.perf_counter()

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

            enforce_rate_limit()  # 🔥 تفعيل نظام الحد من عدد الطلبات

            try:
                if llm:
                    response = agent.invoke({"input": query, "chat_history": []})
                    print_search_results(response)
                else:
                    print("❌ LLM is not available. Please check your setup.")
            except Exception as e:
                logger.error(f"❌ Error during execution: {e}", exc_info=True)
                print("❌ An error occurred during execution.")
    except KeyboardInterrupt:
        print("\n\n👋 Exiting the program. See you next time!")
