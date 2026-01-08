import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from consts import INDEX_NAME
from pinecone import Pinecone

# logger.py는 로그 출력을 예쁘게 하기 위한 사용자 정의 모듈입니다.
from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# OpenAIEmbeddings: 텍스트를 벡터로 변환하는 모델 설정
# text-embedding-3-small: 성능과 비용 면에서 효율적인 최신 모델
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("MY_OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

# Chroma: 로컬에서 실행 가능한 벡터 저장소 (현재 코드에서는 선언만 되고 사용되지 않음)
chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# PineconeVectorStore: 클라우드 기반 벡터 데이터베이스
# index_name: Pinecone 콘솔에서 생성한 인덱스 이름 (consts.py에 정의됨)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# TavilyCrawl: 웹 페이지를 크롤링하여 LLM에 적합한 포맷으로 변환해주는 도구
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """
    문서를 비동기적으로 배치 단위로 나누어 벡터 스토어에 저장합니다.
    대량의 문서를 처리할 때 속도를 높이고, API 호출 제한을 관리하기 위해 사용합니다.
    """
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(
                f"VectorStore Indexing: Failed to add batch {batch_num} - {e}",
                Colors.RED,
            )
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🔍 TavilyCrawl: Starting to crawl documentation from https://docs.langchain.com/oss/python/langchain/",
        Colors.PURPLE,
    )

    # TavilyCrawl.invoke: 설정한 파라미터로 크롤링을 수행합니다.
    # url: 크롤링할 시작 URL
    # extract_depth: "advanced"로 설정하여 깊이 있는 내용 추출
    # max_depth: 링크를 타고 들어갈 깊이 30 (현재 페이지 -> 링크 -> 링크)
    tavily_crawl_results = tavily_crawl.invoke(
        input={
            "url": "https://docs.langchain.com/oss/python/langchain/",
            "extract_depth": "advanced",
            # "instructions": "Documentatin relevant to ai agents",
            "max_depth": 3,
        }
    )
    if tavily_crawl_results.get("error"):
        log_error(f"TavilyCrawl: {tavily_crawl_results['error']}")
        return
    else:
        log_success(
            f"TavilyCrawl: Successfully crawled {len(tavily_crawl_results)} URLs from documentation site"
        )

    all_docs = []
    for tavily_crawl_result_item in tavily_crawl_results["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"] or "",
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    # RecursiveCharacterTextSplitter: 문서를 작은 단위(청크)로 나눕니다.
    # chunk_size=4000: 한 청크당 최대 4000자
    # chunk_overlap=200: 청크 간 200자가 겹치도록 하여 문맥 단절 방지
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Pages crawled: {len(tavily_crawl_results)}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
