from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()


llm = ChatOpenAI(model="gpt-4o")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools = tools)

def main():
    result = agent.invoke({"messages": HumanMessage(content="미국 주식 아이렌의 현재 주가와 최근 컨콜에 대해서 요약해줘")})
    print(result)


if __name__ == "__main__":
    main()
