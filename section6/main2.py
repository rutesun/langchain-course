from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool, BaseTool
from langchain_openai import ChatOpenAI

from section5.callbacks import AgentCallbackHandler

load_dotenv()

"""
section5/main2.py 를 최신 llm 의 tool calling 기능을 사용하여 수정
"""

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    print("Hello LangChain Tools (.bind_tools)!")
    tools = [get_text_length]

    llm = ChatOpenAI(
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )
    llm_with_tools = llm.bind_tools(tools)

    # Start conversation
    messages = [HumanMessage(content="What is the length of the word: DOG")]

    # [Migration Note 1] Prompt Template 제거
    # section5/main2.py 에서는 ReAct 방식의 복잡한 텍스트 프롬프트(Thought/Action/Observation)가 필요했습니다.
    # 하지만 여기서는 LLM의 Native Tool Calling 기능을 사용하므로, 도구 정의만 bind_tools로 넘겨주면 됩니다.
    # 따라서 PromptTemplate 코드가 완전히 사라졌습니다.
    
    while True:
        # [Operation Step 1] LLM 호출
        # 현재까지의 대화 기록(messages)을 LLM에 전달합니다.
        # LLM은 이 기록을 보고 (1) 도구를 호출할지, (2) 최종 답변을 할지 결정합니다.
        ai_message = llm_with_tools.invoke(messages)

        # [Migration Note 2] Parsing 로직 제거 및 안정성 향상
        # 이전에는 LLM이 뱉은 텍스트를 정규식으로 파싱하다가 "Output parsing error"가 자주 발생했습니다.
        # (LLM이 Thought와 Final Answer를 동시에 말해버리는 등)
        # 이제는 ai_message.tool_calls 속성에 구조화된 데이터가 바로 들어오므로 파싱 에러가 원천적으로 차단됩니다.
        
        # If the model decides to call tools, execute them and return results
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        
        # [Operation Step 2] 도구 호출 여부 확인
        if len(tool_calls) > 0:
            # LLM이 도구 사용을 요청했다면, 그 요청(ai_message) 자체를 대화 기록에 추가합니다.
            # 이는 "나는 이런 도구를 쓰려고 생각했어"라는 기록이 됩니다.
            messages.append(ai_message)
            print(f"ai_message={ai_message}")
            
            for tool_call in tool_calls:
                # tool_call is typically a dict with keys: id, type, name, args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                # [Operation Step 3] 도구 실행
                # LLM이 요청한 도구를 실제로 찾아서 실행하고 결과(observation)를 얻습니다.
                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")

                # [Migration Note 3] 수동 루프 (AgentExecutor 대체)
                # section5에서는 AgentExecutor가 내부적으로 루프를 돌며 Action->Observation 과정을 처리했습니다.
                # 여기서는 while True 루프 안에서 직접 ToolMessage를 생성하여 history에 추가하는 방식으로
                # 동작 과정을 투명하게 제어합니다. (LCEL 방식)
                
                # [Operation Step 4] 결과 피드백
                # 도구 실행 결과를 ToolMessage로 포장하여 대화 기록에 추가합니다.
                # tool_call_id를 통해 이 결과가 어떤 도구 호출에 대한 응답인지 명시합니다.
                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            
            # [Operation Step 5] 루프 반복
            # 도구 결과를 추가했으므로, 다시 루프의 처음으로 돌아가 LLM에게 "결과가 이건데, 이제 뭐 할래?"라고 묻습니다.
            # Continue loop to allow the model to use the observations
            continue

        # [Operation Step 6] 최종 답변
        # 도구 호출이 없다면 LLM이 최종 답변을 생성한 것이므로 출력하고 종료합니다.
        # No tool calls -> final answer
        print(ai_message.content)
        break