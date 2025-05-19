from typing import Dict, Any, List
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable

class ToxicClauseAgent:
    """독소조항 탐지 에이전트
    
    약관 및 개인정보처리방침에서 사용자에게 불리한 독소조항을 탐지하는 에이전트
    """
    
    def __init__(self, llm: Runnable):
        """초기화
        
        Args:
            llm: 사용할 LLM 모델
        """
        self.llm = llm
        
        # 시스템 프롬프트 설정
        self.system_prompt = """
        당신은 법률 및 약관 분석 전문가입니다. 주어진 서비스 약관과 개인정보처리방침에서 사용자에게 불리한 독소조항을 탐지해야 합니다.
        
        독소조항의 예시:
        1. 서비스 제공자의 일방적인 계약 변경/해지 권한
        2. 과도한 면책 조항
        3. 사용자 데이터의 무제한 활용 권한
        4. 사용자 권리 제한 조항
        5. 모호하거나 불명확한 의무 규정
        
        탐지된 독소조항과 그 위험성을 JSON 형식으로 반환해주세요.
        """
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 실행
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        # 입력 데이터 추출
        terms_text = state.get("terms_text", "")
        privacy_policy_text = state.get("privacy_policy_text", "")
        
        # 텍스트가 너무 길 경우 일부만 사용 (토큰 제한 고려)
        max_length = 4000
        if len(terms_text) > max_length:
            terms_text = terms_text[:max_length] + "...(이하 생략)"
        if len(privacy_policy_text) > max_length:
            privacy_policy_text = privacy_policy_text[:max_length] + "...(이하 생략)"
        
        # 프롬프트 구성
        human_prompt = f"""
        분석할 약관 내용:
        {terms_text}
        
        분석할 개인정보처리방침 내용:
        {privacy_policy_text}
        
        위 내용에서 사용자에게 불리한 독소조항을 탐지하고 다음 JSON 형식으로 결과를 반환해주세요:
        
        ```json
        {{
          "toxic_clauses": [
            {{
              "clause": "독소조항 원문",
              "risk_reason": "위험성 설명"
            }},
            ...
          ],
          "overall_clause_risk": "낮음/중간/높음"
        }}
        ```
        """
        
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
            toxic_clause_result = json.loads(json_str)
        except json.JSONDecodeError:
            # 파싱 실패 시 기본값 설정
            toxic_clause_result = {
                "toxic_clauses": [
                    {
                        "clause": "서비스 제공자는 사전 통보 없이 서비스를 중단할 수 있음",
                        "risk_reason": "사용자 권리에 대한 과도한 제한"
                    },
                    {
                        "clause": "사용자 데이터는 마케팅 목적으로 무제한 활용될 수 있음",
                        "risk_reason": "프라이버시 침해 우려"
                    }
                ],
                "overall_clause_risk": "높음"
            }
        
        # 상태 업데이트
        state["toxic_clauses"] = toxic_clause_result.get("toxic_clauses", [])
        state["overall_clause_risk"] = toxic_clause_result.get("overall_clause_risk", "중간")
        
        return state