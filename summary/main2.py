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

    agent = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step = agent.invoke(
            {"input": "Eden 사용자의 최근 주문 내역과 배송 상태를 확인해줘.", "intermediate_steps": intermediate_steps},
            config={"run_name": "Order Status Agent (Manual Loop)", "tags": ["order_check", "demo", "manual_loop"]},
        )

        if isinstance(agent_step, AgentAction):
            tool = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(str(tool_input))
            print(f"observation = {observation}")
            intermediate_steps.append((agent_step, str(observation)))

    
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)