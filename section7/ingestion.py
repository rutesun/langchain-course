

import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
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
    loader = TextLoader(file_path)
    document = loader.load()

    print("splitting...")
    char_splitter = CharacterTextSplitter(
        separator="\n\n",     # 구분자 (기본: 문단)
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 청크 최대 크기 (문자)
        chunk_overlap=200,    # 청크 간 중복 (맥락 유지)
        length_function=len,  # 길이 계산 함수
        add_start_index=True  # 원본 위치 추적
    )
    print("splitting... using CharacterTextSplitter")
    texts = char_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    print("splitting... using RecursiveCharacterTextSplitter")
    texts = recur_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")


    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("MY_OPENAI_API_KEY"))

    print("ingesting...")
    vectorstore = Chroma(
        collection_name="my_collection",  # 컬렉션 이름
        embedding_function=embeddings,    # 임베딩 함수
        persist_directory="./chroma_db"   # 로컬 저장 (선택사항)
    )
    vectorstore.add_documents(texts)
    print("finish")