from typing import Dict, Any, List
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable
import textwrap

class ImprovementAgent:
    """개선안 제시 에이전트
    
    윤리적 리스크와 독소조항에 대한 개선 방안을 제시하는 에이전트
    """
    
    def __init__(self, llm: Runnable):
        """초기화
        
        Args:
            llm: 사용할 LLM 모델
        """
        self.llm = llm
        
        # 시스템 프롬프트 설정
        self.system_prompt = """
        당신은 AI 윤리 및 법률 전문가입니다. 주어진 윤리적 리스크와 독소조항에 대한 구체적인 개선 방안을 제시해야 합니다.
        
        각 리스크 항목별로 실행 가능한 개선 방안을 제시하고, 독소조항에 대해서는 공정하고 투명한 대안을 제안해주세요.
        개선 방안은 구체적이고 실용적이어야 합니다.
        
        결과는 JSON 형식으로 반환해주세요.
        """
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 실행
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        # 입력 데이터 추출
        ethical_risks = state.get("ethical_risks", {})
        toxic_clauses = state.get("toxic_clauses", [])
        
        # 리스크 정보 추출
        bias_risk = ethical_risks.get("bias_risk", "알 수 없음")
        privacy_risk = ethical_risks.get("privacy_risk", "알 수 없음")
        explainability_risk = ethical_risks.get("explainability_risk", "알 수 없음")
        automation_risk = ethical_risks.get("automation_risk", "알 수 없음")
        
        # 독소조항 텍스트 추출
        toxic_clause_texts = [clause.get("clause", "") for clause in toxic_clauses]
        
        # 프롬프트 구성
        human_prompt = textwrap.dedent("""
            분석된 윤리적 리스크:
            - 편향성 리스크: {bias_risk}
            - 프라이버시 리스크: {privacy_risk}
            - 설명가능성 리스크: {explainability_risk}
            - 자동화 리스크: {automation_risk}

            탐지된 독소조항:
            {toxic_clauses}

            위 리스크와 독소조항에 대한 구체적인 개선 방안을 다음 JSON 형식으로 제시해주세요:

            ```json
            {{
            "recommendations": {{
                "bias_risk": "편향성 리스크 개선 방안",
                "privacy_risk": "프라이버시 리스크 개선 방안",
                "explainability_risk": "설명가능성 리스크 개선 방안",
                "automation_risk": "자동화 리스크 개선 방안",
                "toxic_clauses": "독소조항 개선 방안"
            }}
            }}
            ```
        """).format(
            bias_risk=bias_risk,
            privacy_risk=privacy_risk,
            explainability_risk=explainability_risk,
            automation_risk=automation_risk,
            toxic_clauses="\n".join([f"- {clause}" for clause in toxic_clause_texts])
        )
        
        # LLM 호출
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = self.llm.invoke(messages)
        
        # 응답에서 JSON 추출
        import json
        import re
        
        # JSON 형식 추출 (간단한 정규식 사용)
        json_match = re.search(r'```json\s*({.*?})\s*```', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content
        
        try:
            # JSON 파싱
            improvement_result = json.loads(json_str)
        except json.JSONDecodeError:
            # 파싱 실패 시 기본값 설정
            improvement_result = {
                "recommendations": {
                    "bias_risk": "사용자 그룹별 테스트 및 결과 로그 분석을 통해 편향 여부 검증",
                    "privacy_risk": "데이터 최소 수집 및 삭제 주기 명시",
                    "explainability_risk": "결과 생성 근거를 사용자에게 제공하는 UI 설계",
                    "automation_risk": "자동화된 추천 결과에 수동 검토 옵션 추가",
                    "toxic_clauses": "약관에서 사용자 권리를 침해하는 조항 제거 또는 명확화"
                }
            }
        
        # 상태 업데이트
        state["recommendations"] = improvement_result.get("recommendations", {})
        
        return state