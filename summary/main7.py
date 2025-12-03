from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import get_order_id, get_order_details

load_dotenv()

if __name__ == "__main__":
    # 1. 도구 및 LLM 설정
    tools = [get_order_id, get_order_details]
    llm = ChatOpenAI(temperature=0)

    # 2. 메모리 설정 (LangGraph의 강력한 기능)
    # 대화의 상태(State)를 저장하여 멀티턴 대화를 가능하게 합니다.
    memory = MemorySaver()

    # 3. 에이전트 생성 (LangGraph 방식)
    # create_react_agent는 컴파일된 그래프(CompiledGraph)를 반환합니다.
    # 이 그래프는 invoke, stream 등의 메서드를 가집니다.
    graph = create_react_agent(model=llm, tools=tools, checkpointer=memory)

    # 4. 실행 설정
    # thread_id를 지정하여 특정 대화 세션을 식별합니다.
    config = {
        "configurable": {"thread_id": "thread-1"},
        "run_name": "Order Status Agent (LangGraph)",
        "tags": ["order_check", "demo", "langgraph"]
    }

    # 5. 실행
    # LangGraph는 입력으로 "messages" 리스트를 받습니다.
    inputs = {"messages": [("user", "Eden 사용자의 최근 주문 내역과 배송 상태를 확인해줘.")]}

    print("--- LangGraph Agent Start ---")
    
    # stream을 사용하여 단계별 실행 과정을 볼 수 있습니다.
    # values에는 현재 상태(messages 등)가 들어있습니다.
    for event in graph.stream(inputs, config=config, stream_mode="values"):
        # 마지막 메시지만 출력해서 진행 상황 확인
        message = event["messages"][-1]
        print(f"[{message.type}]: {message.content}")

    print("--- LangGraph Agent End ---")
