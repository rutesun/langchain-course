from os import name
from dotenv import load_dotenv
from typing import Union, List
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description, tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser

from callbacks import AgentCallbackHandler
from tools import get_order_id, get_order_details, get_shipping_status, find_tool_by_name

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    tools = [get_order_id, get_order_details]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm = ChatOpenAI(temperature=0, stop=["\nObservation", "Observation"], callbacks=[AgentCallbackHandler()])

    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step = agent.invoke(
            {"input": "Eden 사용자의 최근 주문 내역과 배송 상태를 확인해줘.", "agent_scratchpad": intermediate_steps},
            config={"run_name": "Order Status Agent (LCEL)", "tags": ["order_check", "demo", "lcel"]},
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