from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from tools import get_order_id, get_order_details, find_tool_by_name
from callbacks import AgentCallbackHandler
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic import hub

load_dotenv()

if __name__ == "__main__":
    # 1. 도구 및 LLM 설정
    tools = [get_order_id, get_order_details]
    llm = ChatOpenAI(temperature=0)

    # 2. 프롬프트 가져오기 (Tool Calling 전용 프롬프트)
    # "hwchase17/openai-tools-agent"는 시스템 메시지와 placeholder가 잘 설정된 프롬프트입니다.
    prompt = hub.pull("hwchase17/openai-tools-agent")

    # 3. 에이전트 생성 (Native Tool Calling 방식)
    # create_tool_calling_agent:
    # - ReAct 방식(텍스트 파싱)이 아니라, LLM의 Native Tool Calling 기능(bind_tools)을 사용합니다.
    # - 따라서 파싱 에러가 거의 없고 더 안정적입니다.
    # - 내부적으로도 'intermediate_steps'를 사용하지만, 프롬프트 구조가 다릅니다 (MessagesPlaceholder 사용).
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 4. 실행기 생성 (AgentExecutor)
    # AgentExecutor는 에이전트의 종류(ReAct vs Tool Calling)와 상관없이 동일하게 동작합니다.
    # "뇌가 결정을 내리면 -> 몸이 실행하고 -> 기억(intermediate_steps)에 추가한다"는 루프를 관리합니다.
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 5. 실행
    # 여기서도 'intermediate_steps'를 직접 넘기지 않습니다. AgentExecutor가 알아서 관리하기 때문입니다.
    result = agent_executor.invoke(
        {"input": "Eden 사용자의 최근 주문 내역과 배송 상태를 확인해줘."},
        config={"run_name": "Order Status Agent", "tags": ["order_check", "demo", "create_tool_calling_agent", "agentExecutor"]},
    )

    print(result["output"])
