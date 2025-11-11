from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query (str): The query to search for

    Returns:
        str: THe search result
    """
    print(f"Searching for {query}")
    return tavily.search(query = query)

llm = ChatOpenAI(model="gpt-4o")
tools = [search]
agent = create_agent(model=llm, tools = tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="미국 주식 파가야의 현재 주가와 최근 컨톨에 대해서 요약해줘")})
    print(result)


if __name__ == "__main__":
    main()
