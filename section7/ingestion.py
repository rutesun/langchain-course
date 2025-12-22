

import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

"""
https://docs.langchain.com/oss/javascript/langchain/knowledge-base#1-documents-and-document-loaders
https://docs.langchain.com/oss/python/integrations/splitters/index#text-splitters
"""

load_dotenv()

if __name__ == "__main__":
    print("Hello from ingestion!")


    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "mongodb.txt")
    
    # TextLoader: 텍스트 파일을 로드하여 LangChain Document 객체로 변환합니다.
    # 텍스트 파일 뿐만 아니라 PDF, CSV, URL 등 다양한 소스를 로드할 수 있는 로더들이 존재합니다.
    # 예: PyPDFLoader, CSVLoader, UnstructuredLoader 등.
    loader = TextLoader(file_path)
    document = loader.load()

    print("splitting...")
    # CharacterTextSplitter: 지정된 구분자(separator)를 기준으로 텍스트를 나눕니다.
    # 문맥 유지를 위해 chunk_overlap을 사용하여 겹치는 부분을 만듭니다.
    # 단순한 문자열 분리 방식이므로 문장 구조가 깨질 수 있습니다.
    char_splitter = CharacterTextSplitter(
        separator="\n\n",     # 문단 단위로 분리 (기본값)
        chunk_size=1000,      # 청크당 최대 문자 수
        chunk_overlap=200,    # 이전 청크와 겹치는 문자 수 (맥락 유지 용도)
        length_function=len   # 길이 계산 함수 (기본: len)
    )
    
    # RecursiveCharacterTextSplitter: (권장되는 방식)
    # 1. 줄바꿈(\n\n), 2. 줄바꿈(\n), 3. 공백( ), 4. 문자('') 순서로 분리를 시도합니다.
    # 문장이나 단어의 의미적 단위를 최대한 유지하면서 자르기 때문에 검색 성능에 더 좋습니다.
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      
        chunk_overlap=200,    
        length_function=len,  
        add_start_index=True  # 원본 문서에서의 시작 위치 메타데이터 포함
    )

    print("splitting... using CharacterTextSplitter")
    texts = char_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    print("splitting... using RecursiveCharacterTextSplitter")
    texts = recur_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")


    # OpenAIEmbeddings: 텍스트를 벡터(숫자 리스트)로 변환하는 모델입니다.
    # 벡터로 변환하면 단어/문장의 '의미적 유사성'을 계산할 수 있습니다.
    # text-embedding-3-small은 최신 모델로 비교적 저렴하고 성능이 좋습니다.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("MY_OPENAI_API_KEY"), base_url="https://api.openai.com/v1")

    print("ingesting...")
    # VectorStore: 변환된 벡터와 원본 텍스트를 저장하는 데이터베이스입니다.
    # 검색 시 질문 벡터와 가장 유사한(거리가 가까운) 문서 벡터를 찾아서 반환합니다.
    # Chroma(로컬), Pinecone(클라우드), Elasticsearch 등 다양한 구현체가 있습니다.
    
    # 아래는 로컬 ChromaDB 사용 예시:
    # vectorstore = Chroma(
    #     collection_name="my_collection", 
    #     embedding_function=embeddings,    
    #     persist_directory="./chroma_db"   
    # )
    # vectorstore.add_documents(texts)
    
    # Pinecone: 클라우드 기반 관리형 벡터 데이터베이스입니다.
    # from_documents: 문서 리스트를 받아 임베딩 후 인덱스에 저장하는 팩토리 메서드입니다.
    # 이미 인덱스가 존재한다면 해당 인덱스에 데이터를 추가합니다.
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name="langchain"
    )
   
    print("finish")