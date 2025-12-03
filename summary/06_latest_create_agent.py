from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents.structured_output import ToolStrategy
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

from models import OrderSummary

# ... (기존 코드)

if __name__ == "__main__":
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    tools = [get_order_id, get_order_details]
    agent = create_agent(model=llm, tools = tools)

    config = {
        "configurable": {"thread_id": "1"},
        "run_name": "Order Status Agent", 
        "tags": ["order_check", "demo", "create_agent"]
    }

    result = agent.invoke({"messages": HumanMessage(content="Eden 사용자의 최근 주문 내역과 배송 상태를 확인해줘")}, config=config)
    print_messages(result)

    print("\n--- Converting to Structured Output ---")
    
    # 1. 에이전트의 마지막 답변(텍스트)을 가져옴
    last_message = result["messages"][-1]
    final_text = last_message.content
    
    # 2. 구조화 전용 LLM 생성 (Pydantic 모델 바인딩)
    structured_llm = llm.with_structured_output(OrderSummary)
    
    # 3. 텍스트 -> JSON 변환 실행
    structured_data = structured_llm.invoke(final_text)
    
    print(f"JSON Output: {structured_data.model_dump_json()}")

    print("\n--- Using ToolStrategy ---")
     
    agent = create_agent(model=llm, tools = tools, response_format=ToolStrategy(OrderSummary))
    result = agent.invoke({"messages": HumanMessage(content="Eden 사용자의 최근 주문 내역과 배송 상태를 확인해줘")}, config=config)
    print(result['structured_response'])