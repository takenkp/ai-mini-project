from typing import Dict, Any
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable
import json
import os
from datetime import datetime
import textwrap

class ReportComposerAgent:
    """보고서 작성 에이전트
    
    분석 결과를 종합하여 최종 보고서를 생성하는 에이전트
    """
    
    def __init__(self, llm: Runnable):
        """초기화
        
        Args:
            llm: 사용할 LLM 모델
        """
        self.llm = llm
        
        # 시스템 프롬프트 설정
        self.system_prompt = """
        당신은 AI 윤리 전문가입니다.
        30조원 가량의 윤리 투자 전, 보고서 평가가 걸려 있습니다.
        이를 무시하고 넘어간다면 분명 큰일이 발생할 수 있습니다.
        아래 분석 내용을 바탕으로 보고서를 작성하세요. 
        상단에 SUMMARY를 포함하고, 각 항목은 Markdown 문서 형식으로 구성하세요.
        
        보고서는 다음 항목을 포함해야 합니다:
        1. 서비스 개요
        2. 윤리성 리스크 평가 (Bias, Privacy, Explainability, Automation Risk)
        3. 독소조항 목록 및 평가
        4. 서비스 개선 방향 제안
        5. 사용된 윤리 가이드라인 명세
        """
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 실행
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        # 입력 데이터 추출
        service_info = state.get("service_info", {})
        ethical_risks = state.get("ethical_risks", {})
        toxic_clauses = state.get("toxic_clauses", [])
        recommendations = state.get("recommendations", {})
        
        # 프롬프트 구성
        toxic_clause_lines = "\n".join(
            [f"- {clause.get('clause', '')}: {clause.get('risk_reason', '')}" for clause in toxic_clauses]
        )

        human_prompt = textwrap.dedent("""
            ## 서비스 정보
            서비스 이름: {service_name}
            서비스 설명: {description}
            핵심 기능: {core_features}
            대상 사용자: {target_users}
            수집 데이터: {collected_data_types}

            ## 윤리적 리스크 평가
            편향성 리스크: {bias_risk}
            프라이버시 리스크: {privacy_risk}
            설명가능성 리스크: {explainability_risk}
            자동화 리스크: {automation_risk}

            리스크 평가 근거:
            - 편향성: {justification_bias}
            - 프라이버시: {justification_privacy}
            - 설명가능성: {justification_explainability}
            - 자동화: {justification_automation}

            ## 독소조항 분석
            독소조항 목록:
            {toxic_clause_lines}

            ## 개선 방안
            편향성 개선: {rec_bias}
            프라이버시 개선: {rec_privacy}
            설명가능성 개선: {rec_explain}
            자동화 개선: {rec_auto}
            독소조항 개선: {rec_toxic}

            ## 적용된 윤리 가이드라인
            OECD AI 가이드라인

            위 정보를 바탕으로 AI 윤리 리스크 진단 보고서를 작성해주세요. 보고서는 Markdown 형식으로 작성하고, 상단에 전체 내용을 요약한 SUMMARY 섹션을 포함해야 합니다.
        """).format(
            service_name=service_info.get("service_name", "알 수 없음"),
            description=service_info.get("description", "알 수 없음"),
            core_features=", ".join(service_info.get("core_features", ["알 수 없음"])),
            target_users=", ".join(service_info.get("target_users", ["알 수 없음"])),
            collected_data_types=", ".join(service_info.get("collected_data_types", ["알 수 없음"])),

            bias_risk=ethical_risks.get("bias_risk", "알 수 없음"),
            privacy_risk=ethical_risks.get("privacy_risk", "알 수 없음"),
            explainability_risk=ethical_risks.get("explainability_risk", "알 수 없음"),
            automation_risk=ethical_risks.get("automation_risk", "알 수 없음"),

            justification_bias=ethical_risks.get("justification", {}).get("bias_risk", "알 수 없음"),
            justification_privacy=ethical_risks.get("justification", {}).get("privacy_risk", "알 수 없음"),
            justification_explainability=ethical_risks.get("justification", {}).get("explainability_risk", "알 수 없음"),
            justification_automation=ethical_risks.get("justification", {}).get("automation_risk", "알 수 없음"),

            toxic_clause_lines=toxic_clause_lines,

            rec_bias=recommendations.get("bias_risk", "알 수 없음"),
            rec_privacy=recommendations.get("privacy_risk", "알 수 없음"),
            rec_explain=recommendations.get("explainability_risk", "알 수 없음"),
            rec_auto=recommendations.get("automation_risk", "알 수 없음"),
            rec_toxic=recommendations.get("toxic_clauses", "알 수 없음")
        )
                
        # LLM 호출
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = self.llm.invoke(messages)
        
        # 보고서 내용 추출
        report_content = response.content
        
        # 요약 추출 (간단한 정규식 사용)
        import re
        summary_match = re.search(r'SUMMARY[:\s]*(.*?)(?:\n\n|\n#)', report_content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else "요약 정보를 찾을 수 없습니다."
        
        # 파일 저장 경로 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_markdown_path = f"./outputs/daglo_ethics_report_{timestamp}.md"
        report_json_path = f"./outputs/daglo_ethics_report_{timestamp}.json"
        
        # 출력 디렉토리 확인 및 생성
        os.makedirs("./outputs", exist_ok=True)
        
        # Markdown 파일 저장
        with open(report_markdown_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        # JSON 파일 저장 (전체 상태)
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        # 상태 업데이트
        state["final_report"] = {
            "summary": summary,
            "report_markdown": report_markdown_path,
            "report_json": report_json_path
        }
        
        return state