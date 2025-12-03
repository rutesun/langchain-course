from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field
from tools import get_order_id, get_order_details, find_tool_by_name

load_dotenv()


def print_messages(result):
    messages = result["messages"]
    for msg in messages:
        # 1. 메시지 타입에 따라 헤더 출력
        role = msg.type.upper()
        print(f"\n[{role}]")
        
        # 2. 내용 출력
        if msg.content:
            print(f"Content: {msg.content}")
            
        # 3. 도구 호출 정보 출력 (AI 메시지인 경우)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"Tool Call: {tool_call['name']}({tool_call['args']})")
                
        print("-" * 50)

if __name__ == "__main__":
    llm = ChatOpenAI(temperature=0)
    tools = [get_order_id, get_order_details]
    agent = create_agent(model=llm, tools = tools)

    result = agent.invoke({"messages": HumanMessage(content="Eden 사용자의 최근 주문 내역과 배송 상태를 확인해줘")})
    print_messages(result)
