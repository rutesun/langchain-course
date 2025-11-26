from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search Top 5 news in USA stock market today"
                )
            ]
        }
    )
    print(result)
    
if __name__ == "__main__":  
    main()