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

    # [ReAct 프롬프트 템플릿]
    # ReAct 에이전트의 핵심은 "Thought(생각) -> Action(행동) -> Observation(관찰)"의 루프를 가르치는 것입니다.
    # {tools}: 사용 가능한 도구들의 설명이 들어갑니다.
    # {tool_names}: 도구들의 이름 목록이 들어갑니다.
    # {agent_scratchpad}: 이전 단계의 생각과 행동 결과(History)가 여기에 쌓입니다.
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

    # partial: 템플릿의 변수 중 일부(tools, tool_names)를 미리 채워넣습니다.
    # 실행 시점에는 input과 agent_scratchpad만 넣으면 됩니다.
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    
    # stop: LLM이 스스로 "Observation:"을 생성하지 못하게 막습니다. (관찰은 도구의 실행 결과여야 하므로)
    llm = ChatOpenAI(temperature=0, stop=["\nObservation", "Observation"], callbacks=[AgentCallbackHandler()])

    intermediate_steps = []

    # [LCEL 체인 구성]
    # 1. 입력 변수 매핑: input은 그대로, agent_scratchpad는 포맷팅하여 전달
    # 2. prompt: 완성된 프롬프트 생성
    # 3. llm: LLM 실행
    # 4. ReActSingleInputOutputParser: LLM의 텍스트 출력을 AgentAction 또는 AgentFinish 객체로 파싱
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # [수동 실행 루프]
    # AgentExecutor를 쓰지 않고 직접 루프를 돌리는 방식입니다.
    # 동작 원리를 이해하는 데 매우 중요합니다.
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        # 1. 에이전트(뇌)에게 현재 상황을 주고 다음 행동을 결정하게 함
        agent_step = agent.invoke(
            {"input": "Eden 사용자의 최근 주문 내역과 배송 상태를 확인해줘.", "agent_scratchpad": intermediate_steps},
            config={"run_name": "Order Status Agent (LCEL)", "tags": ["order_check", "demo", "lcel"]},
        )

        # 2. 에이전트가 "행동(Action)"을 하기로 결정했다면?
        if isinstance(agent_step, AgentAction):
            tool = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool)
            tool_input = agent_step.tool_input
            
            # 3. 도구 실행 (Observation 획득)
            observation = tool_to_use.func(str(tool_input))
            print(f"observation = {observation}")
            
            # 4. 결과 기록 (기억 추가)
            # (Action, Observation) 쌍을 intermediate_steps에 추가하여 다음 턴에 LLM에게 보여줍니다.
            intermediate_steps.append((agent_step, str(observation)))

    
    # 5. 최종 답변(Finish) 도달 시 출력
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)