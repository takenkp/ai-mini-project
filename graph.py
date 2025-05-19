from typing import Dict, Any, TypedDict, List
from langgraph.graph import StateGraph, END
from agents.service_analysis_agent import ServiceAnalysisAgent
from agents.ethical_risk_agent import EthicalRiskAgent
from agents.toxic_clause_agent import ToxicClauseAgent
from agents.improvement_agent import ImprovementAgent
from agents.report_composer_agent import ReportComposerAgent

# 상태 타입 정의
class State(TypedDict, total=False):
    service_url: str
    documents: List[str]
    terms_text: str
    privacy_policy_text: str
    service_info: Dict[str, Any]
    ethical_risks: Dict[str, Any]
    toxic_clauses: List[Dict[str, str]]
    recommendations: Dict[str, Any]
    final_report: Dict[str, Any]

# 그래프 빌더 함수
def build_ethics_assessment_graph(llm):
    """AI 윤리 리스크 진단 워크플로우 그래프 생성
    
    Args:
        llm: 사용할 LLM 모델
        
    Returns:
        컴파일된 StateGraph 객체
    """
    # 에이전트 초기화
    service_analysis_agent = ServiceAnalysisAgent(llm=llm)
    ethical_risk_agent = EthicalRiskAgent(llm=llm)
    toxic_clause_agent = ToxicClauseAgent(llm=llm)
    improvement_agent = ImprovementAgent(llm=llm)
    report_composer_agent = ReportComposerAgent(llm=llm)
    
    # 상태 그래프 생성
    workflow = StateGraph(State)
    
    # 노드 추가
    workflow.add_node("service_analysis", service_analysis_agent)
    workflow.add_node("ethical_risk", ethical_risk_agent)
    workflow.add_node("toxic_clause", toxic_clause_agent)
    workflow.add_node("improvement", improvement_agent)
    workflow.add_node("report_composer", report_composer_agent)
    
    # 엣지 추가 (순차적 플로우)
    workflow.add_edge("service_analysis", "ethical_risk")
    workflow.add_edge("ethical_risk", "toxic_clause")
    workflow.add_edge("toxic_clause", "improvement")
    workflow.add_edge("improvement", "report_composer")
    workflow.add_edge("report_composer", END)
    
    # 그래프 시각화 (선택적)
    # workflow.show()
    
    # 컴파일 및 반환
    return workflow.compile()