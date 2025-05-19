import os
import json
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from agents.service_analysis_agent import ServiceAnalysisAgent
from agents.ethical_risk_agent import EthicalRiskAgent
from agents.toxic_clause_agent import ToxicClauseAgent
from agents.improvement_agent import ImprovementAgent
from agents.report_composer_agent import ReportComposerAgent
from dotenv import load_dotenv

# 환경 변수 설정 (실제 사용 시 .env 파일 또는 환경 변수로 설정)
load_dotenv()

# 상태 타입 정의
StateType = Dict[str, Any]

# 기본 LLM 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 에이전트 초기화
service_analysis_agent = ServiceAnalysisAgent(llm=llm)
ethical_risk_agent = EthicalRiskAgent(llm=llm)
toxic_clause_agent = ToxicClauseAgent(llm=llm)
improvement_agent = ImprovementAgent(llm=llm)
report_composer_agent = ReportComposerAgent(llm=llm)

# 그래프 정의
def build_graph():
    # 상태 그래프 생성
    workflow = StateGraph(StateType)
    
    # 노드 추가
    workflow.set_entry_point("service_analysis")

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
    
    # 컴파일
    return workflow.compile()

# 메인 실행 함수
def run_ethics_assessment(service_url: str, documents: List[str], terms_text: str, privacy_policy_text: str):
    # 그래프 빌드
    graph = build_graph()
    
    # 초기 상태 설정
    initial_state = {
        "service_url": service_url,
        "documents": documents,
        "terms_text": terms_text,
        "privacy_policy_text": privacy_policy_text,
        "service_info": {},
        "ethical_risks": {},
        "toxic_clauses": [],
        "recommendations": {},
        "final_report": {}
    }
    
    # 그래프 실행
    result = graph.invoke(initial_state)
    
    # 결과 반환
    return result

# 예제 실행 코드
if __name__ == "__main__":
    # 예제 입력 데이터
    service_url = "https://daglo.ai/guide"
    documents = ["./data/daglo_guide.pdf"]
    
    # 약관 및 개인정보처리방침 (실제로는 파일에서 읽어오거나 웹에서 가져옴)
    with open("./data/daglo_terms.txt", "r", encoding="utf-8") as f:
        terms_text = f.read()
    
    privacy_policy_text = terms_text  # 예제에서는 동일한 텍스트 사용
    
    # 윤리 평가 실행
    result = run_ethics_assessment(
        service_url=service_url,
        documents=documents,
        terms_text=terms_text,
        privacy_policy_text=privacy_policy_text
    )
    
    # 결과 출력
    print("\n===== AI 윤리 리스크 진단 결과 =====")
    print(f"요약: {result['final_report'].get('summary', '요약 정보 없음')}")
    print(f"보고서 경로: {result['final_report'].get('report_markdown', '보고서 경로 없음')}")
    
    # JSON 결과 저장
    with open(result['final_report'].get('report_json', './outputs/result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("\n평가가 완료되었습니다. 자세한 내용은 생성된 보고서를 확인하세요.")