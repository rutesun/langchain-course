from os import name
from langchain_classic import hub
from dotenv import load_dotenv
from typing import Union, List
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description, tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from callbacks import AgentCallbackHandler
from tools import get_order_id, get_order_details, get_shipping_status, find_tool_by_name

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    tools = [get_order_id, get_order_details]

    react_prompt = hub.pull("hwchase17/react")
    
    llm = ChatOpenAI(temperature=0, callbacks=[AgentCallbackHandler()])
    agent = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)

    # AgentExecutor: 에이전트의 뇌(agent)와 몸통(tools)을 결합하여 실행 루프를 자동화합니다.
    # [동작 원리]
    # 1. AgentExecutor는 내부적으로 while 루프를 돕니다.
    # 2. 첫 실행 시 'intermediate_steps'는 빈 리스트([])로 시작합니다.
    # 3. Agent(뇌)가 도구 사용(Action)을 결정하면, Executor가 대신 도구를 실행하고 결과(Observation)를 얻습니다.
    # 4. 이 결과(Action, Observation) 쌍을 'intermediate_steps' 리스트에 추가합니다.
    # 5. 다시 Agent를 호출할 때, 업데이트된 'intermediate_steps'를 자동으로 주입해줍니다.
    #    (따라서 개발자가 수동으로 리스트를 관리하거나 넘겨줄 필요가 없습니다.)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    result = agent_executor.invoke(
        {"input": "Eden 사용자의 최근 주문 내역과 배송 상태를 확인해줘."},
        # config: LangSmith 추적을 위한 메타데이터 설정
        config={"run_name": "Order Status Agent", "tags": ["order_check", "demo"]},
    )

    print(result["output"])