"""
작성자 : kp
작성일 : 2025-05-18 (수정: 2025-05-20)
목적 : Chroma + BM25 기반의 EnsembleRetriever 구성 및 검색 결과 출처 표기
내용 : 지정된 PDF 디렉토리와 Chroma DB 경로를 사용하여 EnsembleRetriever를 생성.
       HuggingFaceEmbeddings 임포트 경로를 LangChain 0.2.2+ 권장 사항에 맞게 수정.
"""

import os
import glob
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # LangChain 0.2.2+
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# 기본 설정값 (주로 직접 실행 시 또는 기본값으로 사용)
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

def load_and_split_documents_for_bm25(
    pdf_dir: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """BM25 Retriever를 위한 문서를 로드하고 청킹합니다."""
    docs_for_bm25 = []
    if not os.path.exists(pdf_dir) or not os.path.isdir(pdf_dir):
        print(f"⚠️  경고 (BM25): PDF 디렉토리 '{pdf_dir}'를 찾을 수 없습니다.")
        return docs_for_bm25
        
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        print(f"ℹ️  정보 (BM25): PDF 디렉토리 '{pdf_dir}' 내에 PDF 파일이 없습니다.")
        return docs_for_bm25

    raw_docs = []
    print(f"📄 (BM25용) '{pdf_dir}'에서 총 {len(pdf_files)}개 PDF 파일 로드 중...")
    for pdf_path in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_path)
            raw_docs.extend(loader.load())
        except Exception as e:
            print(f"❌ 에러 (BM25): '{os.path.basename(pdf_path)}' 로드 중 오류 발생: {e}")

    if not raw_docs:
        return docs_for_bm25

    print(f"✂️  (BM25용) 문서 청크 분할 중 (크기: {chunk_size}, 중첩: {chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""], # 다양한 구분자 사용
        length_function=len,
    )
    docs_for_bm25 = splitter.split_documents(raw_docs)
    print(f"  (BM25용) {len(docs_for_bm25)}개 청크 생성 완료.")
    return docs_for_bm25


def build_ensemble_retriever(
    pdf_dir: str, # RAG 대상 문서가 있는 디렉토리
    chroma_persist_dir: str, # 해당 서비스의 ChromaDB 저장 경로
    k_results: int = 3, # 가져올 검색 결과 수
    bm25_weight: float = 0.4, # BM25 가중치
    chroma_weight: float = 0.6, # Chroma 가중치
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    chunk_size_for_bm25: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap_for_bm25: int = DEFAULT_CHUNK_OVERLAP
) -> Optional[EnsembleRetriever]:
    """
    지정된 경로의 문서와 ChromaDB를 사용하여 EnsembleRetriever를 생성합니다.

    Args:
        pdf_dir: BM25 및 원문 참조를 위한 PDF 문서가 있는 디렉토리.
        chroma_persist_dir: ChromaDB 데이터가 저장된/저장될 디렉토리.
        k_results: 검색 시 반환할 결과의 수.
        bm25_weight: EnsembleRetriever에서 BM25 결과의 가중치.
        chroma_weight: EnsembleRetriever에서 Chroma 결과의 가중치.
        embedding_model_name: 사용할 임베딩 모델 이름.
        chunk_size_for_bm25: BM25용 문서 청킹 시 크기.
        chunk_overlap_for_bm25: BM25용 문서 청킹 시 중첩 크기.

    Returns:
        구성된 EnsembleRetriever 객체 또는 실패 시 None.
    """
    print(f"🧠 HuggingFace 임베딩 로딩 (모델: {embedding_model_name})...")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    except Exception as e:
        print(f"❌ 에러: 임베딩 모델 '{embedding_model_name}' 로드 중 오류 발생: {e}")
        print("   (HINT: `pip install -U langchain-huggingface`를 실행했는지 확인하세요.)")
        return None

    print(f"🔍 Chroma 벡터 저장소 로딩 중 (경로: {chroma_persist_dir})...")
    if not os.path.exists(chroma_persist_dir) or not os.listdir(chroma_persist_dir):
        print(f"❌ 에러: Chroma DB 디렉토리 '{chroma_persist_dir}'가 비어 있거나 존재하지 않습니다.")
        print(f"   먼저 '{pdf_dir}'의 문서를 해당 Chroma 경로로 인덱싱해야 합니다.")
        return None
        
    try:
        chroma_vectorstore = Chroma(
            persist_directory=chroma_persist_dir,
            embedding_function=embedding_model,
        )
    except Exception as e:
        print(f"❌ 에러: Chroma DB ('{chroma_persist_dir}') 로드 중 오류 발생: {e}")
        return None

    semantic_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": k_results})
    print("  Chroma 리트리버 준비 완료.")

    print(f"📄 BM25 리트리버용 원문 로딩 및 구축 중 (소스: {pdf_dir})...")
    bm25_docs = load_and_split_documents_for_bm25(
        pdf_dir,
        chunk_size=chunk_size_for_bm25,
        chunk_overlap=chunk_overlap_for_bm25
    )
    
    lexical_retriever = None
    if not bm25_docs:
        print("⚠️  경고: BM25 리트리버를 위한 문서가 없어 BM25는 제외하고 Chroma 리트리버만 사용합니다.")
    else:
        try:
            lexical_retriever = BM25Retriever.from_documents(bm25_docs)
            lexical_retriever.k = k_results
            print("  BM25 리트리버 준비 완료.")
        except Exception as e:
            print(f"❌ 에러: BM25 리트리버 생성 중 오류 발생: {e}")

    if lexical_retriever and semantic_retriever:
        print("🔗 EnsembleRetriever 구성 중 (Chroma + BM25)...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, lexical_retriever],
            weights=[chroma_weight, bm25_weight],
        )
    elif semantic_retriever:
        print("🔗 Chroma 리트리버만 사용 (BM25 생성 실패 또는 문서 없음).")
        ensemble_retriever = semantic_retriever # BM25가 없으면 Chroma만 사용
    else:
        print("❌ 에러: Semantic retriever (Chroma)도 준비되지 않았습니다. Retriever를 구성할 수 없습니다.")
        return None


    print("✅ 리트리버 구성 완료.")
    return ensemble_retriever

if __name__ == "__main__":
    print("--- Ensemble Retriever 직접 실행 테스트 ---")
    
    # 이 테스트는 특정 서비스의 데이터 디렉토리와 Chroma DB 경로를 지정해야 합니다.
    # 예: test_pdf_dir = "./data/daglo"
    # 예: test_chroma_dir = "./vectorstore/chroma_daglo"
    # 위 경로에 대해 indexer.py가 먼저 실행되어 DB가 생성되어 있어야 합니다.

    test_pdf_dir = "./data/daglo" # 테스트할 PDF 문서가 있는 폴더
    service_name_for_db = os.path.basename(test_pdf_dir) # 'daglo'
    test_chroma_dir = f"./vectorstore/chroma_{service_name_for_db}" # 테스트할 Chroma DB 경로

    print(f"테스트 대상 PDF 폴더: {test_pdf_dir}")
    print(f"테스트 대상 Chroma DB 폴더: {test_chroma_dir}")

    if not (os.path.exists(test_pdf_dir) and os.path.exists(test_chroma_dir) and os.listdir(test_chroma_dir)):
        print(f"\n🚫 테스트 중단: 테스트를 위한 PDF 폴더 또는 Chroma DB가 준비되지 않았습니다.")
        print(f"   '{test_pdf_dir}'에 PDF 파일이 있는지,")
        print(f"   '{test_chroma_dir}'에 해당 PDF에 대한 인덱싱된 Chroma DB가 있는지 확인하세요.")
        print(f"   (HINT: python indexing/indexer.py --service_data_dir {test_pdf_dir} 와 같이 실행하여 먼저 인덱싱하세요.)")
    else:
        retriever = build_ensemble_retriever(
            pdf_dir=test_pdf_dir,
            chroma_persist_dir=test_chroma_dir,
            k_results=3
        )

        if retriever:
            query = "Daglo AI Guide의 주요 기능은 무엇인가?"
            print(f"\n💬 검색어: \"{query}\"")
            
            try:
                results = retriever.invoke(query) 
            except AttributeError: 
                 results = retriever.get_relevant_documents(query)

            print(f"\n📌 검색 결과 ({len(results)}개):")
            if not results:
                print("  검색된 문서가 없습니다.")
                
            for i, doc in enumerate(results):
                content_preview = doc.page_content[:200].replace("\n", " ") + "..."
                source_file = doc.metadata.get('source_file', doc.metadata.get('source', 'N/A'))
                page_num = doc.metadata.get('page', 'N/A') 
                section_title = doc.metadata.get('section_title', 'N/A')

                print(f"\n--- 결과 {i + 1} ---")
                print(f"  📖 내용 (일부): \"{content_preview}\"")
                print(f"  📄 출처 파일: {source_file}")
                print(f"  - 페이지: {page_num if page_num != 'N/A' else '정보 없음'} (0-based)")
                print(f"  - 추론된 섹션: {section_title if section_title != 'N/A' else '정보 없음'}")
        else:
            print("🚫 리트리버 생성에 실패하여 테스트를 진행할 수 없습니다.")
    
    print("\n--- 테스트 종료 ---")
