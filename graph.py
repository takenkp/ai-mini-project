from typing import Dict, Any, TypedDict, List, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.retrievers.ensemble import EnsembleRetriever 

from agents.service_analysis_agent import ServiceAnalysisAgent
from agents.ethical_risk_agent import EthicalRiskAgent 
from agents.toxic_clause_agent import ToxicClauseAgent 
from agents.improvement_agent import ImprovementAgent # 수정된 버전 임포트
from agents.report_composer_agent import ReportComposerAgent # 수정된 버전 임포트

MAX_JOIN_ATTEMPTS = 5 

class State(TypedDict, total=False):
    service_url: Optional[str]
    documents: List[str] 
    
    service_info: Dict[str, Any]
    
    ethical_risks: Dict[str, Any]
    toxic_clauses: List[Dict[str, str]] 
    overall_clause_risk: str 

    recommendations: Dict[str, Any]
    final_report: Dict[str, Any]

    ethical_risk_done: bool
    toxic_clause_done: bool
    join_attempt_count: int 
    error_message: Optional[str]


def build_ethics_assessment_graph(
        llm: ChatOpenAI, 
        retriever_instance: EnsembleRetriever | None,
        guideline_keyword_for_ethics: str = "OECD",
        report_output_dir: str = "./outputs" # ReportComposerAgent용 출력 디렉토리
    ):
    print(f"그래프 빌드 시작 (병렬, 가이드라인 키워드: {guideline_keyword_for_ethics}, 보고서 출력: {report_output_dir})...")
    prompt_directory = "./prompts" 

    service_analysis_agent = ServiceAnalysisAgent(llm=llm, retriever=retriever_instance, prompt_dir=prompt_directory)
    ethical_risk_agent = EthicalRiskAgent(
        llm=llm, 
        retriever=retriever_instance, 
        guideline_doc_keyword=guideline_keyword_for_ethics,
        prompt_dir=prompt_directory
    )
    toxic_clause_agent = ToxicClauseAgent(llm=llm, retriever=retriever_instance, prompt_dir=prompt_directory)
    improvement_agent = ImprovementAgent(llm=llm, prompt_dir=prompt_directory)
    # ReportComposerAgent에 output_dir 전달
    report_composer_agent = ReportComposerAgent(llm=llm, prompt_dir=prompt_directory, output_dir=report_output_dir) 
    
    workflow = StateGraph(State)
    
    # --- 노드 정의 ---
    def service_analysis_node(state: State) -> Dict[str, Any]:
        print("노드: service_analysis 실행...")
        try:
            return service_analysis_agent(state)
        except Exception as e:
            print(f"오류: service_analysis_node에서 예외 발생 - {e}")
            return {"error_message": f"Service Analysis 실패: {str(e)}"}

    def ethical_risk_node(state: State) -> Dict[str, Any]:
        print("노드: ethical_risk_assessment 실행...")
        if state.get("error_message"): return {"ethical_risk_done": True} 
        try:
            result = ethical_risk_agent(state) 
            return {**result, "ethical_risk_done": True}
        except Exception as e:
            print(f"오류: ethical_risk_node에서 예외 발생 - {e}")
            return {"error_message": state.get("error_message","") + f"; Ethical Risk Assessment 실패: {str(e)}", "ethical_risk_done": True}

    def toxic_clause_node(state: State) -> Dict[str, Any]:
        print("노드: toxic_clause_detection 실행...")
        if state.get("error_message"): return {"toxic_clause_done": True} 
        try:
            result = toxic_clause_agent(state) 
            return {**result, "toxic_clause_done": True}
        except Exception as e:
            print(f"오류: toxic_clause_node에서 예외 발생 - {e}")
            return {"error_message": state.get("error_message","") + f"; Toxic Clause Detection 실패: {str(e)}", "toxic_clause_done": True}

    def join_for_improvement_node(state: State) -> Dict[str, Any]:
        print("노드: join_for_improvement 실행...")
        if state.get("error_message") and not (state.get("ethical_risk_done") and state.get("toxic_clause_done")):
             print(f"  이전 단계 오류로 인해 Join 중단: {state.get('error_message')}")
             return {} 
        current_attempts = state.get("join_attempt_count", 0) + 1
        print(f"  Join 시도 횟수: {current_attempts}")
        return {"join_attempt_count": current_attempts}

    def improvement_node(state: State) -> Dict[str, Any]:
        print("노드: improvement_generation 실행...")
        if state.get("error_message"): return {}
        try:
            return improvement_agent(state)
        except Exception as e:
            print(f"오류: improvement_node에서 예외 발생 - {e}")
            return {"error_message": state.get("error_message","") + f"; Improvement Generation 실패: {str(e)}"}

    def report_node(state: State) -> Dict[str, Any]:
        print("노드: report_composition 실행...")
        # ReportComposerAgent가 내부적으로 오류를 처리하고 final_report에 상태를 기록함
        return report_composer_agent(state)

    def handle_fatal_error_node(state: State) -> Dict[str, Any]:
        print("노드: handle_fatal_error 실행...")
        error_msg = state.get("error_message", "알 수 없는 심각한 오류 발생")
        print(f"  치명적 오류 처리: {error_msg}")
        final_report_error = {
            "summary": "진단 프로세스 중 심각한 오류 발생",
            "error_details": error_msg,
            "status": "Failed"
        }
        return {"final_report": final_report_error} 

    workflow.add_node("service_analysis", service_analysis_node)
    workflow.add_node("ethical_risk_assessment", ethical_risk_node)
    workflow.add_node("toxic_clause_detection", toxic_clause_node)
    workflow.add_node("join_for_improvement", join_for_improvement_node)
    workflow.add_node("improvement_generation", improvement_node)
    workflow.add_node("report_composition", report_node)
    workflow.add_node("handle_fatal_error", handle_fatal_error_node)

    workflow.set_entry_point("service_analysis")

    def check_service_analysis_error(state: State) -> str:
        if state.get("error_message"):
            print(f"  Service Analysis 후 오류 감지: {state.get('error_message')}")
            return "fatal_error_branch"
        print("  Service Analysis 성공. 병렬 브랜치로 진행.")
        return "continue_to_parallel_branches"
    workflow.add_node("start_parallel_tasks_dummy_node", lambda state: state) # 아무 작업 안 함
    workflow.add_conditional_edges(
        "service_analysis",
        check_service_analysis_error,
        {
            "fatal_error_branch": "handle_fatal_error",
            "continue_to_parallel_branches": "start_parallel_tasks_dummy_node"
        }
    )
    # 더미 노드에서 각 병렬 브랜치로 엣지 추가
    workflow.add_edge("start_parallel_tasks_dummy_node", "ethical_risk_assessment")
    workflow.add_edge("start_parallel_tasks_dummy_node", "toxic_clause_detection")

    workflow.add_edge("ethical_risk_assessment", "join_for_improvement")
    workflow.add_edge("toxic_clause_detection", "join_for_improvement")

    def decide_after_join(state: State) -> str:
        # join 노드 이전에 발생한 오류(예: 병렬 작업 중 하나가 error_message 설정) 확인
        if state.get("error_message") and ("Ethical Risk Assessment 실패" in state.get("error_message") or \
                                           "Toxic Clause Detection 실패" in state.get("error_message") ):
            print(f"  병렬 작업 중 오류 발생 감지 (Join 후): {state.get('error_message')}")
            return "fatal_error_after_join" 

        ethical_done = state.get("ethical_risk_done", False)
        toxic_done = state.get("toxic_clause_done", False)
        attempts = state.get("join_attempt_count", 0)
        print(f"Join 후 진행 조건 확인: Ethical Done={ethical_done}, Toxic Done={toxic_done}, 시도={attempts}")

        if ethical_done and toxic_done:
            print("  모든 병렬 분석 완료. Improvement 생성으로 진행.")
            return "proceed_to_improvement"
        elif attempts >= MAX_JOIN_ATTEMPTS:
            print(f"  최대 Join 시도 횟수 ({MAX_JOIN_ATTEMPTS}회) 초과. 타임아웃.")
            # 타임아웃 시, 상태에 오류 메시지 설정하고 오류 처리 노드로
            # join_for_improvement_node에서 attempt_count만 증가시키므로, 여기서 error_message 설정
            state["error_message"] = (state.get("error_message") or "") + "; 병렬 작업(Ethical Risk, Toxic Clause) 완료 대기 중 타임아웃 발생."
            return "timeout_error" 
        else:
            print("  아직 모든 병렬 작업 완료되지 않음. Join 재시도 (루프).")
            return "retry_join"

    workflow.add_conditional_edges(
        "join_for_improvement",
        decide_after_join,
        {
            "proceed_to_improvement": "improvement_generation",
            "retry_join": "join_for_improvement", 
            "timeout_error": "handle_fatal_error", 
            "fatal_error_after_join": "handle_fatal_error" 
        }
    )
    
    def check_improvement_error(state: State) -> str:
        if state.get("error_message") and "Improvement Generation 실패" in state.get("error_message",""):
            print(f"  Improvement 생성 후 오류 감지: {state.get('error_message')}")
            return "fatal_error_after_improvement"
        print("  Improvement 생성 성공. 보고서 작성으로 진행.")
        return "continue_to_report"

    workflow.add_conditional_edges(
        "improvement_generation",
        check_improvement_error,
        {
            "fatal_error_after_improvement": "handle_fatal_error",
            "continue_to_report": "report_composition"
        }
    )
    
    # report_composition 노드는 내부적으로 오류를 처리하고 final_report에 상태를 기록하므로,
    # 여기서 별도의 오류 분기 없이 바로 END로 연결.
    workflow.add_edge("report_composition", END)
    workflow.add_edge("handle_fatal_error", END) 
    
    print("그래프 컴파일 중...")
    compiled_graph = workflow.compile()
    print("그래프 빌드 및 컴파일 완료.")
    return compiled_graph
