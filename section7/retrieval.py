import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == "__main__":
    print(" Retrieving...")

    # OpenAIEmbeddings: 질문 텍스트를 벡터로 변환하기 위해 필요합니다.
    # ingestion 시 사용했던 모델과 *반드시* 동일해야 합니다. (같은 공간 상에 매핑되기 위함)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("MY_OPENAI_API_KEY"), base_url="https://api.openai.com/v1")
    
    # LLM: 답변을 생성할 언어 모델입니다.
    llm = ChatOpenAI()

    query = "어떤 요소가 이번 MongoDb 의 Q3 2026 Earnings Call에서 제일 중요했을 것 같아? 답변은 한글로 해줘"
    
    # PromptTemplate: LLM에게 보낼 프롬프트 형식을 정의합니다.
    # 단순한 질문-답변 체인을 테스트하는 코드(주석 처리됨).
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    # PineconeVectorStore: 기존에 저장된 인덱스("langchain")를 불러옵니다.
    # 데이터를 추가하는 것이 아니라, 검색(Retrieval)을 위해 객체를 생성합니다.
    vectorstore = PineconeVectorStore(
        index_name="langchain", embedding=embeddings
    )

    # LangChain Hub: 검증된 프롬프트 템플릿들을 공유하고 다운로드 받을 수 있는 저장소입니다.
    # "langchain-ai/retrieval-qa-chat"은 RAG(Retrieval-Augmented Generation)를 위한 표준 프롬프트입니다.
    # 내용은 "다음 컨텍스트를 사용하여 질문에 답하시오" 같은 형태입니다.
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # create_stuff_documents_chain: "Stuff" 방식의 문서 결합 체인입니다.
    # 검색된 모든 문서(Documents)의 내용을 단순히 이어붙여서(Stuff) 프롬프트의 {context} 변수에 넣습니다.
    # 가장 단순하지만 토큰 제한에 걸릴 수 있습니다. (문서가 너무 많으면 잘릴 수 있음)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    # create_retrieval_chain: 검색기(Retriever)와 문서 결합 체인(Combine Docs Chain)을 연결합니다.
    # 동작 과정:
    # 1. 사용자의 질문(input)을 받음
    # 2. retriever가 관련된 문서를 검색
    # 3. 검색된 문서를 combine_docs_chain으로 전달
    # 4. LLM이 답변 생성
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    # chain.invoke({"input": ...}) 형태로 실행하며, 내부적으로는 문서를 찾아오고 답변을 생성합니다.
    result = retrival_chain.invoke(input={"input": query})

    print(result)