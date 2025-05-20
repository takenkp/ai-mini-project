"""
작성자 : kp
작성일 : 2025-05-18 (수정: 2025-05-19)
목적 : PDF 문서 청크 및 벡터 임베딩 후 ChromaDB 저장
내용 : PyMuPDFLoader + RecursiveCharacterTextSplitter + HuggingFaceEmbeddings 기반으로
       PDF 문서를 읽고 폰트 크기 기반으로 추론된 섹션 제목을 포함하여 chroma에 저장
"""

import os
import glob
from typing import List, Dict, Any
import shutil
import fitz # PyMuPDF

# Langchain 라이브러리 임포트 (환경에 따라 langchain_community 등으로 변경될 수 있음)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 📁 설정
PDF_DIR = "./data/claude/"  # indexer.py 파일 위치 기준 상대 경로
CHROMA_DIR = "./vectorstore/chroma_claude" # indexer.py 파일 위치 기준 상대 경로
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
# 폰트 기반 제목 추론을 위한 임계값
TITLE_FONT_SIZE_MIN_DIFFERENCE = 1.5 # 일반 텍스트보다 최소 이만큼 커야 제목으로 간주 (절대값)
TITLE_FONT_SIZE_MIN_RATIO = 1.15    # 일반 텍스트보다 최소 이 비율만큼 커야 제목으로 간주 (비율)


def extract_section_title_by_font_heuristic(page: fitz.Page) -> str:
    """
    주어진 fitz.Page 객체에서 폰트 크기 기반 휴리스틱을 사용하여 섹션 제목을 추론합니다.
    가장 큰 폰트 크기를 가진 텍스트를 제목으로 간주하되, 일반 텍스트와 충분히 구분될 때만 인정합니다.
    """
    font_counts: Dict[float, int] = {} # 폰트 크기별 문자 수 카운트
    text_spans_by_size: Dict[float, List[Dict[str, Any]]] = {} # 폰트 크기별 텍스트 스팬 저장

    blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
    if not blocks:
        return "N/A"

    for block in blocks:
        if block["type"] == 0:  # Text block
            for line in block["lines"]:
                for span in line["spans"]:
                    size = round(span["size"], 2) # 소수점 둘째 자리까지 반올림하여 유사 폰트 그룹화
                    text = span["text"].strip()
                    if not text: # 빈 텍스트는 무시
                        continue
                    
                    font_counts[size] = font_counts.get(size, 0) + len(text)
                    
                    if size not in text_spans_by_size:
                        text_spans_by_size[size] = []
                    text_spans_by_size[size].append({
                        "text": text,
                        "y": span["bbox"][1], # 정렬을 위한 y 좌표
                        "x": span["bbox"][0]  # 정렬을 위한 x 좌표
                    })

    if not font_counts:
        return "N/A"

    # 가장 흔한 폰트 크기를 본문 텍스트 크기로 간주 (문자 수 기준)
    sorted_font_counts = sorted(font_counts.items(), key=lambda item: item[1], reverse=True)
    body_text_size = sorted_font_counts[0][0] if sorted_font_counts else 0.0

    # 페이지에서 가장 큰 폰트 크기 찾기
    largest_font_size_on_page = 0.0
    if font_counts:
        largest_font_size_on_page = max(font_counts.keys())
    
    # 제목으로 간주할 수 있는지 여부 판단
    # 1. 페이지에 다양한 폰트 크기가 사용되었고,
    # 2. 가장 큰 폰트가 본문 폰트보다 의미있게 클 때
    is_title_plausible = (
        len(font_counts) > 1 and
        largest_font_size_on_page > body_text_size and
        (largest_font_size_on_page >= body_text_size + TITLE_FONT_SIZE_MIN_DIFFERENCE or
         largest_font_size_on_page >= body_text_size * TITLE_FONT_SIZE_MIN_RATIO)
    )

    if is_title_plausible and largest_font_size_on_page in text_spans_by_size:
        title_spans = text_spans_by_size[largest_font_size_on_page]
        # y, x 좌표 순으로 정렬하여 텍스트 결합
        title_spans.sort(key=lambda s: (s["y"], s["x"]))
        page_section_title = " ".join(s["text"] for s in title_spans if s["text"])
        
        # 제목 길이 제한 (너무 길면 자르기)
        if len(page_section_title) > 250:
            page_section_title = page_section_title[:250] + "..."
        return page_section_title
    
    return "N/A" # 그 외의 경우 제목을 찾지 못함


def load_documents_from_dir(pdf_dir: str) -> List[Dict[str, Any]]: # Langchain Document는 Dict[str, Any]로 표현될 수 있음
    """폴더 내 모든 PDF 문서를 불러오고 폰트 크기 기반으로 추론된 섹션 제목을 메타데이터에 추가"""
    all_docs_with_metadata = []
    if not os.path.exists(pdf_dir) or not os.path.isdir(pdf_dir):
        print(f"⚠️  경고: PDF 디렉토리 '{pdf_dir}'를 찾을 수 없거나 디렉토리가 아닙니다.")
        return all_docs_with_metadata

    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        print(f"ℹ️  정보: PDF 디렉토리 '{pdf_dir}' 내에 PDF 파일이 없습니다.")
        return all_docs_with_metadata

    print(f"📂 총 {len(pdf_files)}개의 PDF 파일 감지.")
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"\n📄 '{file_name}' 로드 및 처리 중...")
        
        fitz_doc = None # fitz.Document 객체 초기화
        try:
            fitz_doc = fitz.open(pdf_path)
            
            # Langchain의 PyMuPDFLoader를 사용하여 페이지별 Document 객체 생성
            loader = PyMuPDFLoader(pdf_path)
            docs_from_loader = loader.load() # 페이지별 Document 객체 리스트
            
            for i, lc_doc_page in enumerate(docs_from_loader):
                lc_doc_page.metadata['source_file'] = file_name # 파일명 메타데이터 추가
                
                if i < len(fitz_doc):
                    fitz_page = fitz_doc[i] # 현재 페이지의 fitz.Page 객체
                    section_title = extract_section_title_by_font_heuristic(fitz_page)
                    lc_doc_page.metadata['section_title'] = section_title
                else:
                    # PyMuPDFLoader가 로드한 페이지 수와 fitz_doc의 페이지 수가 다를 경우 방어 코드
                    lc_doc_page.metadata['section_title'] = "N/A (페이지 인덱스 불일치)"
                    
            all_docs_with_metadata.extend(docs_from_loader)
            print(f"  '{file_name}' 로드 완료. (페이지 수: {len(docs_from_loader)})")

        except Exception as e:
            print(f"❌ 에러: '{file_name}' 처리 중 오류 발생: {e}")
        finally:
            if fitz_doc:
                fitz_doc.close() # fitz.Document 객체 닫기
            
    return all_docs_with_metadata


def split_documents(
    docs: List[Dict[str, Any]], # Langchain Document
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]: # Langchain Document
    """문서를 청크 단위로 분할 (메타데이터는 상속됨)"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""], # 다양한 구분자 사용
        length_function=len, # 문자열 길이 계산 함수
    )
    chunked_docs = splitter.split_documents(docs)
    print(f"✂️  총 {len(docs)}개 원본 페이지(문서)를 {len(chunked_docs)}개의 청크로 분할 완료.")
    return chunked_docs


def print_chunking_examples(chunked_docs: List[Dict[str, Any]], num_examples: int = 3, preview_length: int = 100):
    """분할된 청크의 예시를 출력"""
    print(f"\n🔍 청킹 예시 (처음 {num_examples}개 청크 미리보기):")
    if not chunked_docs:
        print("  (보여줄 청크가 없습니다.)")
        return

    for i, chunk in enumerate(chunked_docs[:num_examples]):
        source_file = chunk.metadata.get('source_file', chunk.metadata.get('source', 'N/A'))
        page_number = chunk.metadata.get('page', 'N/A') # PyMuPDFLoader가 0-based로 추가
        section_title = chunk.metadata.get('section_title', 'N/A')
        
        content_preview_raw = chunk.page_content[:preview_length]
        content_preview_processed = content_preview_raw.replace('\n', ' ')
        
        print(f"\n  --- 청크 {i+1} ---")
        print(f"  출처 파일: {source_file}")
        print(f"  페이지 번호 (0-based): {page_number}")
        print(f"  추론된 섹션 제목: {section_title}")
        print(f"  내용 (처음 {preview_length}자):")
        print(f"    \"{content_preview_processed}...\"")
        print(f"  (청크 길이: {len(chunk.page_content)}자)")


def index_documents(docs: List[Dict[str, Any]], persist_dir: str):
    """문서를 임베딩하고 Chroma에 저장"""
    try:
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            # encode_kwargs={'normalize_embeddings': True} # 필요시 코사인 유사도 위해 정규화
        )
    except Exception as e:
        print(f"❌ 에러: 임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로드 중 오류 발생: {e}")
        return

    print(f"✅ 문서 {len(docs)}개 임베딩 시작 (모델: {EMBEDDING_MODEL_NAME})...")
    try:
        vectorstore = Chroma.from_documents(
            documents=docs, # Langchain Document 객체 리스트
            embedding=embedding,
            persist_directory=persist_dir,
        )
        vectorstore.persist() # 변경사항 디스크에 즉시 저장
        print(f"✅ 벡터 DB 저장 완료: {persist_dir}")
    except Exception as e:
        print(f"❌ 에러: 문서 임베딩 또는 Chroma DB 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    print("--- PDF 임베딩 프로세스 시작 (폰트 크기 기반 섹션 추론) ---")

    print("\n📦 PDF 로드 및 메타데이터(섹션 제목) 추출 중...")
    raw_docs_with_metadata = load_documents_from_dir(PDF_DIR)

    if not raw_docs_with_metadata:
        print("🚫 로드된 PDF 문서가 없어 프로세스를 중단합니다.")
    else:
        print(f"\n✂️  문서 청크 분할 중 (청크 크기: {CHUNK_SIZE}, 중첩: {CHUNK_OVERLAP})...")
        chunked_docs = split_documents(raw_docs_with_metadata)

        print_chunking_examples(chunked_docs) # 청킹 결과 예시 출력

        print("\n💾 벡터 DB 저장 준비 중...")
        # Chroma 디렉토리 존재 및 데이터 유무 확인
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            choice = input(f"⚠️  경고: 벡터 DB 디렉토리 '{CHROMA_DIR}'에 이미 데이터가 존재합니다. \n    기존 데이터를 삭제하고 새로 생성하시겠습니까? (y/n): ").strip().lower()
            if choice == 'y':
                print(f"🗑️  기존 벡터 DB '{CHROMA_DIR}' 삭제 중...")
                shutil.rmtree(CHROMA_DIR) # 디렉토리와 내용 모두 삭제
                os.makedirs(CHROMA_DIR, exist_ok=True) # 삭제 후 다시 생성
                print("🧠 임베딩 및 저장 시작...")
                index_documents(chunked_docs, CHROMA_DIR)
            else:
                print("🚫 작업을 중단합니다. 기존 DB를 유지합니다.")
        else:
            os.makedirs(CHROMA_DIR, exist_ok=True) # 디렉토리가 없으면 생성
            print("🧠 임베딩 및 저장 시작...")
            index_documents(chunked_docs, CHROMA_DIR)

    print("\n--- 모든 프로세스 완료 ---")
