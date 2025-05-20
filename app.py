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
        return {"error": f"Service data directory not found: {service_data_dir}", "final_report": {"status": "Setup Error"}}

    service_pdf_paths = glob.glob(os.path.join(service_data_dir, "*.pdf"))
    if not service_pdf_paths:
        print(f"경고: '{service_data_dir}' 내에 분석할 서비스 PDF 문서가 없습니다.")

    all_document_paths = list(set(service_pdf_paths + (guideline_doc_paths if guideline_doc_paths else [])))
    if not all_document_paths:
         print(f"경고: 분석할 PDF 문서(서비스 또는 가이드라인)가 전혀 없습니다.")
    
    service_url_to_analyze = service_url if service_url is not None else ""
    if not service_url_to_analyze and not all_document_paths:
        print("오류: 서비스 URL 또는 분석 대상 PDF 문서(서비스 또는 가이드라인) 중 하나 이상은 제공되어야 합니다.")
        return {"error": "Insufficient input for analysis.", "final_report": {"status": "Input Error"}}
        
    print("LLM 초기화 중 (gpt-4o)...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, request_timeout=120, max_retries=2)

    service_name_for_db = os.path.basename(os.path.normpath(service_data_dir))
    chroma_persist_dir = os.path.join("./vectorstore", f"chroma_{service_name_for_db}") 
    os.makedirs(os.path.dirname(chroma_persist_dir), exist_ok=True)

    print(f"Retriever 초기화 중 (k={retriever_k_results}, PDF 소스: {service_data_dir} 및 가이드라인, Chroma DB: {chroma_persist_dir})...")
    retriever_instance = None
    if all_document_paths: 
        try:
            retriever_instance = build_ensemble_retriever(
                pdf_dir=service_data_dir, 
                chroma_persist_dir=chroma_persist_dir,
                k_results=retriever_k_results
            )
            if retriever_instance is None:
                print(f"경고: Retriever 초기화 실패. '{service_data_dir}'에 대한 인덱싱이 필요할 수 있습니다.")
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
    except FileNotFoundError as e: 
        print(f"오류: 그래프 빌드 실패 (필수 프롬프트 파일 누락 가능성) - {e}")
        return {"error": f"Graph build failed due to missing file: {e}", "final_report": {"status": "Build Error"}}
    except Exception as e:
        print(f"오류: 그래프 빌드 중 예기치 않은 오류 발생 - {e}")
        return {"error": f"Unexpected error during graph build: {e}", "final_report": {"status": "Build Error"}}

    if graph is None: 
        return {"error": "Graph compilation failed.", "final_report": {"status": "Build Error"}}

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
        # 실행 중 오류 발생 시 final_state가 None일 수 있으므로, 오류 상태를 만들어 반환
        final_state = initial_state # 최소한 초기 상태라도 사용
        final_state["error_message"] = f"Graph execution error: {str(e)}"
        final_state["final_report"] = { # final_report 필드도 초기화
            "summary": "진단 프로세스 실행 중 오류 발생",
            "error_details": f"Graph execution error: {str(e)}",
            "status": "Execution Error"
        }

    print("진단 워크플로우 실행 완료.")
    
    if final_state is None: # 만약의 경우를 대비한 방어 코드
        print("오류: 그래프 실행 후 최종 상태가 없습니다.")
        return {"error": "Graph execution resulted in no final state.", "final_report": {"status": "Execution Error"}}

    # 최종 상태 전체를 output_dir에 저장
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_service_name = os.path.basename(os.path.normpath(service_data_dir))
        # 최종 상태 JSON 파일명에 타임스탬프 추가 (덮어쓰기 방지)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_state_json_path = os.path.join(output_dir, f"ethics_assessment_final_state_{output_service_name}_{timestamp}.json")
        with open(final_state_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_state, f, ensure_ascii=False, indent=2, default=str) 
        print(f"전체 최종 상태 (JSON): {os.path.abspath(final_state_json_path)}")
    except Exception as e:
        print(f"오류: 최종 상태 JSON 저장 실패 - {e}")

    # 화면에 요약 및 보고서 경로 출력
    print("\n===== AI 윤리 리스크 진단 결과 요약 =====")
    if final_state.get("error_message"): 
        print(f"오류로 인해 파이프라인이 정상적으로 완료되지 못했습니다: {final_state.get('error_message')}")
    
    final_report_info = final_state.get("final_report", {}) # final_state가 None이 아님을 위에서 보장
    if isinstance(final_report_info, dict):
        print(f"요약: {final_report_info.get('summary', '요약 정보 없음 (또는 오류 발생)')}")
        
        report_md_path = final_report_info.get('report_markdown') 
        if report_md_path and os.path.exists(report_md_path): 
             print(f"보고서 (Markdown): {os.path.abspath(report_md_path)}")
        elif report_md_path: # 경로 정보는 있으나 파일이 없을 경우
            print(f"보고서 (Markdown): 파일이 생성되지 않았거나 경로가 잘못되었습니다 - {report_md_path}")
        else: # 경로 정보 자체가 없을 경우 (예: 오류로 생성 안됨)
            print(f"보고서 (Markdown): 생성되지 않음")

        report_pdf_path = final_report_info.get('report_pdf') 
        if report_pdf_path and os.path.exists(report_pdf_path):
            print(f"보고서 (PDF): {os.path.abspath(report_pdf_path)}")
        elif report_pdf_path:
            print(f"보고서 (PDF): 파일이 생성되지 않았거나 경로가 잘못되었습니다 - {report_pdf_path}")
        else:
            print(f"보고서 (PDF): 생성되지 않음 (오류 또는 변환 실패)")
            
        # 상태 코드를 통해 좀 더 명확한 피드백 제공
        report_status = final_report_info.get('status', '알 수 없음')
        print(f"진단 상태: {report_status}")
        if report_status not in ["Success", "Partial Success (PDF Convert Failed)"]:
            print(f"오류 상세: {final_report_info.get('error_details', final_report_info.get('error', '상세 정보 없음'))}")
    else:
        print("최종 보고서 정보를 찾을 수 없거나 형식이 올바르지 않습니다.")

    print("\n평가가 완료되었습니다. 자세한 내용은 생성된 보고서를 확인하세요.")
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

if __name__ == "__main__":
    main()
