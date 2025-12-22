import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# ============================================================================
# 초기화 단계
# ============================================================================
print("Initializing components...")


embeddings = OpenAIEmbeddings(api_key=os.getenv("MY_OPENAI_API_KEY"), base_url="https://api.openai.com/v1")
llm = ChatOpenAI()

# Pinecone 연결: 이미 ingestion.py를 통해 데이터가 저장되어 있어야 합니다.
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# Retriever: VectorStore를 검색기 인터페이스로 변환합니다.
# search_kwargs={"k": 3} -> 가장 유사한 문서 3개를 가져오라는 의미입니다.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ChatPromptTemplate: 대화형 모델을 위한 프롬프트 템플릿입니다.
# {context}: 검색된 문서 내용이 들어갈 자리
# {question}: 사용자 질문이 들어갈 자리
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """검색된 문서(Document 객체 리스트)들을 하나의 문자열로 합치는 헬퍼 함수입니다."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    LCEL(LangChain Expression Language)을 사용하지 않은 수동 구현 방식입니다.
    RAG(Retrieval-Augmented Generation)의 내부 동작 원리를 이해하기 좋습니다.
    
    단점:
    - 코드가 길어지고 복잡해질 수 있음
    - 스트리밍, 비동기 처리를 직접 구현해야 함
    """
    # 1단계: 문서 검색 (Retrieve)
    # 벡터 DB에서 쿼리와 유사한 문서를 가져옵니다.
    docs = retriever.invoke(query)

    # 2단계: 컨텍스트 포맷팅 (Format)
    # 문서 내용을 하나의 문자열로 만듭니다.
    context = format_docs(docs)

    # 3단계: 프롬프트 생성 (Prompting)
    # 템플릿에 컨텍스트와 질문을 채워넣습니다.
    messages = prompt_template.format_messages(context=context, question=query)

    # 4단계: LLM 호출 (Generation)
    # 완성된 프롬프트를 LLM에 전달하여 답변을 얻습니다.
    response = llm.invoke(messages)

    # 5단계: 내용 반환
    return response.content


# ============================================================================
# 구현 2: LCEL (LangChain Expression Language) 사용 - 권장되는 방식
# ============================================================================
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

def create_retrieval_chain_with_lcel():
    """
    LCEL을 사용한 체인 생성 함수입니다.
    파이프(|) 연산자를 사용하여 데이터의 흐름을 선언적으로 정의합니다.
    
    장점:
    - 가독성: 데이터 흐름이 명확함 (Input -> Retriever -> Prompt -> LLM -> Output)
    - 기능: 스트리밍(.stream()), 비동기(.ainvoke()), 배치(.batch()) 자동 지원
    - 확장성: 다른 체인과 결합하기 쉬움
    """

    # 디버깅 팁:
    # 1. langchain.globals.set_debug(True): 모든 단계의 입출력을 상세히 로깅합니다.
    # 2. LangSmith 사용: 웹 UI에서 트레이스를 시각적으로 확인하는 가장 강력한 방법입니다.
    # 3. 단계별 분리 테스트: 체인의 일부만 끊어서 실행해볼 수 있습니다.
    #    예: (retriever | format_docs).invoke("질문")

    # RunnablePassthrough.assign: 입력 데이터(딕셔너리)에 새로운 키-값을 추가합니다.
    # 여기서는 "context" 키에 (질문 -> 검색 -> 포맷팅) 결과를 할당합니다.
    # itemgetter("question"): 입력 딕셔너리에서 "question" 값을 가져옵니다.
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template  # 딕셔너리(question, context 포함)가 프롬프트 템플릿으로 전달됨
        | llm              # 프롬프트 메시지가 LLM으로 전달됨
        | StrOutputParser() # LLM의 응답(AIMessage)을 문자열로 변환
    )
    return retrieval_chain  

if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "이번 MongoDb 의 Q3 2026 Earnings Call에서 가장 중요한 핵심 내용 5가지를 요약해줘. 왜 매출이 예상보다 높았는지도 설명해줘. 답변은 한글로 해줘"

    # ========================================================================
    # 옵션 0: RAG 없이 생으로 질문하기
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    # 배경지식(Context) 없이 모델이 학습한 데이터로만 답변하므로, 최신 정보나 비공개 정보를 모릅니다.
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # 옵션 1: LCEL 없이 수동 RAG 구현
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # 옵션 2: LCEL을 사용한 RAG 구현 (권장)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- 더 간결하고 명확한 코드 구조")
    print("- 자동 지원: 스트리밍 chain.stream(), 비동기 chain.ainvoke()")
    print("- 디버깅 용이성: LangSmith 등과 연동 시 각 단계 추적 가능")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()

    query_result = RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    ).invoke({"question": query})
    print(query_result)
    print()
    
    # invoke 시 딕셔너리 형태로 질문을 전달해야 합니다. (itemgetter가 키를 참조하므로)
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
