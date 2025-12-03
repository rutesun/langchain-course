from os import name
from langchain_classic import hub
from dotenv import load_dotenv
from typing import Union, List
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description, tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent
from callbacks import AgentCallbackHandler
from tools import get_order_id, get_order_details, get_shipping_status, find_tool_by_name

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    tools = [get_order_id, get_order_details]

    react_prompt = hub.pull("hwchase17/react")
    
    llm = ChatOpenAI(temperature=0, stop=["\nObservation", "Observation"], callbacks=[AgentCallbackHandler()])

    intermediate_steps = []

    # [create_react_agent]
    # 이전 파일(01_react_manual_lcel.py)에서 수동으로 구성했던 LCEL 체인을 한 번에 만들어주는 함수입니다.
    # 내부적으로는 똑같이 (prompt | llm | parser) 구조를 가집니다.
    agent = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)
    
    agent_step = ""
    
    # [수동 실행 루프]
    # 01번 파일과 동일한 루프입니다.
    # AgentExecutor를 사용하면 이 부분이 자동화되지만, 
    # 여기서는 "에이전트가 어떻게 생각하고 행동하는지"를 단계별로 확인하기 위해 수동으로 돌립니다.
    while not isinstance(agent_step, AgentFinish):
        # 1. 에이전트 호출
        # create_react_agent로 만든 에이전트는 "intermediate_steps"라는 이름으로 기록을 받습니다.
        agent_step = agent.invoke(
            {"input": "Eden 사용자의 최근 주문 내역과 배송 상태를 확인해줘.", "intermediate_steps": intermediate_steps},
            config={"run_name": "Order Status Agent (Manual Loop)", "tags": ["order_check", "demo", "manual_loop"]},
        )

        # 2. 행동(Action) 처리
        if isinstance(agent_step, AgentAction):
            tool = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool)
            tool_input = agent_step.tool_input
            
            # 3. 도구 실행
            observation = tool_to_use.func(str(tool_input))
            print(f"observation = {observation}")
            
            # 4. 기록 업데이트
            intermediate_steps.append((agent_step, str(observation)))

    
    # 5. 종료 처리
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)