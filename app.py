import os
import json
from typing import Dict, List, Any, Optional
import argparse
import glob

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from graph import build_ethics_assessment_graph, State 
from indexing.retriever import build_ensemble_retriever 

load_dotenv()

def run_ethics_assessment_pipeline(
    service_data_dir: str, 
    guideline_doc_paths: Optional[List[str]] = None, 
    service_url: Optional[str] = None,
    retriever_k_results: int = 3,
    output_dir: str = "./outputs",
    guideline_keyword: str = "OECD" 
    ):
    print(f"AI 윤리 리스크 진단 파이프라인 시작 (대상 서비스 폴더: {service_data_dir})...")

    if not os.path.exists(service_data_dir) or not os.path.isdir(service_data_dir):
        print(f"오류: 서비스 데이터 디렉토리 '{service_data_dir}'를 찾을 수 없습니다.")
        return {"error": f"Service data directory not found: {service_data_dir}"}

    service_pdf_paths = glob.glob(os.path.join(service_data_dir, "*.pdf"))
    if not service_pdf_paths:
        print(f"경고: '{service_data_dir}' 내에 분석할 서비스 PDF 문서가 없습니다.")

    all_document_paths = list(set(service_pdf_paths + (guideline_doc_paths if guideline_doc_paths else [])))
    if not all_document_paths:
            print(f"경고: 분석할 PDF 문서(서비스 또는 가이드라인)가 전혀 없습니다.")
    
    print(f"DEBUG: app.py - all_document_paths for RAG: {all_document_paths}") # <--- 추가된 로그

    service_url_to_analyze = service_url if service_url is not None else ""
    if not service_url_to_analyze and not all_document_paths:
        print("오류: 서비스 URL 또는 분석 대상 PDF 문서(서비스 또는 가이드라인) 중 하나 이상은 제공되어야 합니다.")
        return {"error": "Insufficient input for analysis."}
        
    print("LLM 초기화 중 (gpt-4o)...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, request_timeout=120, max_retries=2)

    service_name_for_db = os.path.basename(os.path.normpath(service_data_dir))
    chroma_persist_dir = os.path.join("./vectorstore", f"chroma_{service_name_for_db}") 
    os.makedirs(os.path.dirname(chroma_persist_dir), exist_ok=True)

    print(f"Retriever 초기화 중 (k={retriever_k_results}, PDF 소스: {service_data_dir} 및 가이드라인, Chroma DB: {chroma_persist_dir})...")
    retriever_instance = None
    
    # RAG 대상이 될 PDF 문서가 하나라도 있을 때 Retriever 초기화 시도
    if all_document_paths: 
        print(f"DEBUG: app.py - Attempting to call build_ensemble_retriever for PDF_DIR: {service_data_dir}, CHROMA_DIR: {chroma_persist_dir}") # <--- 추가된 로그
        try:
            retriever_instance = build_ensemble_retriever(
                pdf_dir=service_data_dir, 
                chroma_persist_dir=chroma_persist_dir,
                k_results=retriever_k_results
                # embedding_model_name, chunk_size 등은 retriever.py의 기본값 사용
            )
            print(f"DEBUG: app.py - build_ensemble_retriever call finished. Retriever instance type: {type(retriever_instance)}") # <--- 추가된 로그
            if retriever_instance is None:
                print(f"경고: Retriever 초기화 실패 (Chroma DB '{chroma_persist_dir}'가 준비되지 않았거나 {service_data_dir}에 PDF가 없을 수 있습니다).")
                print(f"      먼저 'python indexing/indexer.py --service_data_dir {service_data_dir}'를 실행하여 인덱싱하세요.")
        except Exception as e:
            print(f"오류: Retriever 생성 중 예외 발생 - {e}. RAG 기능이 제한될 수 있습니다.")
    else:
        print(f"정보: 분석할 PDF 문서가 없어 Retriever를 초기화하지 않습니다.")

    print("진단 워크플로우 그래프 빌드 중...")
    graph = None
    try:
        graph = build_ethics_assessment_graph(
            llm=llm, 
            retriever_instance=retriever_instance,
            guideline_keyword_for_ethics=guideline_keyword,
            report_output_dir=output_dir
        )
        print(graph.get_graph().draw_mermaid())

    except FileNotFoundError as e: 
        print(f"오류: 그래프 빌드 실패 (필수 프롬프트 파일 누락 가능성) - {e}")
        return {"error": f"Graph build failed due to missing file: {e}"}
    except Exception as e:
        print(f"오류: 그래프 빌드 중 예기치 않은 오류 발생 - {e}")
        return {"error": f"Unexpected error during graph build: {e}"}

    if graph is None: 
        return {"error": "Graph compilation failed."}

    initial_state: State = {
        "service_url": service_url_to_analyze,
        "documents": all_document_paths, 
        "service_info": {}, 
        "ethical_risks": {}, 
        "toxic_clauses": [], 
        "overall_clause_risk": "", 
        "recommendations": {}, 
        "final_report": {},
        "ethical_risk_done": False, 
        "toxic_clause_done": False,
        "join_attempt_count": 0, 
        "error_message": None 
    }
    print(f"초기 상태 설정 완료: URL='{service_url_to_analyze}', 전체 문서 수={len(all_document_paths)}")
    
    print("진단 워크플로우 실행 시작...")
    final_state = None
    try:
        final_state = graph.invoke(initial_state, config={'recursion_limit': 150})
    except Exception as e:
        print(f"오류: 그래프 실행 중 예외 발생 - {e}")
        final_state = initial_state 
        final_state["error_message"] = f"Graph execution error: {str(e)}"
        final_state["final_report"] = {
            "summary": "진단 프로세스 실행 중 오류 발생",
            "error_details": f"Graph execution error: {str(e)}",
            "status": "Execution Error"
        }

    print("진단 워크플로우 실행 완료.")
    
    if final_state is None:
        print("오류: 그래프 실행 후 최종 상태가 없습니다.")
        return {"error": "Graph execution resulted in no final state."}

    final_report_info = final_state.get("final_report", {})
    if isinstance(final_report_info, dict):
        report_md_path_from_agent = final_report_info.get('report_markdown') 
        report_json_path_from_agent = final_report_info.get('report_json')

        if report_md_path_from_agent:
            abs_md_path = os.path.abspath(os.path.join(output_dir, os.path.basename(report_md_path_from_agent)))
            final_state["final_report"]["report_markdown_abs"] = abs_md_path
        if report_json_path_from_agent:
            abs_json_path = os.path.abspath(os.path.join(output_dir, os.path.basename(report_json_path_from_agent)))
            final_state["final_report"]["report_json_abs"] = abs_json_path
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_service_name = os.path.basename(os.path.normpath(service_data_dir))
            final_state_json_path = os.path.join(output_dir, f"ethics_assessment_final_state_{output_service_name}.json")
            with open(final_state_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_state, f, ensure_ascii=False, indent=2, default=str) 
            print(f"전체 최종 상태 (JSON): {os.path.abspath(final_state_json_path)}")

        except TypeError as te:
            print(f"오류: 최종 상태 JSON 직렬화 실패 - {te}.")
            try:
                fallback_json_path = os.path.join(output_dir, f"ethics_assessment_report_only_{output_service_name}.json")
                with open(fallback_json_path, 'w', encoding='utf-8') as f:
                    json.dump(final_report_info, f, ensure_ascii=False, indent=2, default=str)
                print(f"  (대안으로 보고서 정보만 저장 시도: {os.path.abspath(fallback_json_path)})")
            except Exception as e_fallback:
                print(f"  대안 저장도 실패: {e_fallback}")
        except Exception as e:
            print(f"오류: JSON 결과 저장 실패 - {e}")

    elif final_state.get("error_message"): 
        print(f"파이프라인 오류: {final_state.get('error_message')}")
    else:
        print("오류: 최종 보고서 정보가 올바른 형식이 아닙니다.")
    return final_state


def main():
    parser = argparse.ArgumentParser(description="AI 윤리 리스크 진단 파이프라인 실행 도구")
    parser.add_argument("--service_data_dir", type=str, required=True, 
                        help="분석 대상 서비스의 문서(PDF)가 포함된 디렉토리 경로. 윤리 가이드라인 PDF도 이 폴더에 함께 위치해야 RAG 대상이 됩니다.")
    parser.add_argument("--guideline_docs", nargs="*", default=[],
                        help="참고할 윤리 가이드라인 PDF 문서 경로 목록 (선택 사항, 공백으로 구분). service_data_dir과 다른 경로에 있을 경우 지정. (현재 Retriever는 service_data_dir만 사용하므로, 가이드라인 문서도 해당 폴더에 위치시키는 것을 권장)")
    parser.add_argument("--guideline_keyword", type=str, default="OECD",
                        help="EthicalRiskAgent가 RAG 쿼리 시 참조할 가이드라인 문서의 키워드 (기본값: OECD).")
    parser.add_argument("--url", type=str, default=None,
                        help="분석 대상 서비스의 URL (선택 사항).")
    parser.add_argument("--k_results", type=int, default=3, 
                        help="RAG 검색 시 가져올 문서 청크 수 (기본값: 3).")
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                        help="결과 보고서 및 JSON 파일을 저장할 디렉토리 (기본값: ./outputs).")

    args = parser.parse_args()
    
    guideline_absolute_paths = [os.path.abspath(p) for p in args.guideline_docs] if args.guideline_docs else []

    result_state = run_ethics_assessment_pipeline(
        service_data_dir=os.path.abspath(args.service_data_dir), 
        guideline_doc_paths=guideline_absolute_paths, 
        service_url=args.url,
        retriever_k_results=args.k_results,
        output_dir=os.path.abspath(args.output_dir), 
        guideline_keyword=args.guideline_keyword
    )
    
    print("\n===== AI 윤리 리스크 진단 결과 요약 =====")
    if result_state.get("error_message"): 
        print(f"오류로 인해 파이프라인이 정상적으로 완료되지 못했습니다: {result_state.get('error_message')}")
    
    final_report_info = result_state.get("final_report", {})
    if isinstance(final_report_info, dict):
        print(f"요약: {final_report_info.get('summary', '요약 정보 없음 (또는 오류 발생)')}")
        report_md_path = final_report_info.get('report_markdown_abs', final_report_info.get('report_markdown', '보고서 경로 없음'))
        if os.path.exists(str(report_md_path)): 
                print(f"보고서 (Markdown): {report_md_path}")
        else:
            print(f"보고서 (Markdown): 경로를 찾을 수 없거나 생성되지 않음 - {report_md_path}")
            
        if final_report_info.get("status") == "Error" or final_report_info.get("status") == "Failed" or final_report_info.get("status") == "Execution Error":
            print(f"오류 상세: {final_report_info.get('error_details', final_report_info.get('error', '상세 정보 없음'))}")

    else:
        print("최종 보고서 정보를 찾을 수 없거나 형식이 올바르지 않습니다.")

    print("\n평가가 완료되었습니다. 자세한 내용은 생성된 보고서를 확인하세요.")

if __name__ == "__main__":
    main()
