from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from tools import get_order_id, get_order_details, find_tool_by_name
from callbacks import AgentCallbackHandler

load_dotenv()

if __name__ == "__main__":
    # 1. 도구 설정
    tools = [get_order_id, get_order_details]

    # 2. LLM 설정 (bind_tools 사용)
    # ReAct 프롬프트 대신, LLM에게 "너는 이런 도구들을 쓸 수 있어"라고 직접 알려줍니다.
    llm = ChatOpenAI(temperature=0, callbacks=[AgentCallbackHandler()])
    llm_with_tools = llm.bind_tools(tools)

    # 3. 대화 시작
    messages = [HumanMessage(content="Eden 사용자의 최근 주문 내역과 배송 상태를 확인해줘.")]

    print("Start conversation with bind_tools...")
    
    while True:
        # 4. LLM 호출 (config로 LangSmith 설정 추가)
        ai_message = llm_with_tools.invoke(
            messages,
            config={"run_name": "Order Status Agent (Bind Tools)", "tags": ["order_check", "demo", "bind_tools"]}
        )

        # 5. 도구 호출 여부 확인
        # ai_message.tool_calls 속성에 구조화된 도구 호출 정보가 들어있습니다.
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        
        if len(tool_calls) > 0:
            print(f"LLM wants to call tools: {len(tool_calls)} calls")
            # LLM의 요청(ai_message)을 대화 기록에 추가
            messages.append(ai_message)
            
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                # 도구 실행
                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation({tool_name}) = {observation}")

                # 결과 피드백 (ToolMessage)
                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            
            # 도구 결과를 가지고 다시 루프 (LLM에게 결과 전달)
            continue

        # 6. 최종 답변
        print("Final Answer:")
        print(ai_message.content)
        break
