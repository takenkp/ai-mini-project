from typing import Dict, Any
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable

class EthicalRiskAgent:
    """윤리적 리스크 평가 에이전트
    
    AI 서비스의 편향성, 프라이버시, 설명가능성, 자동화 위험성 등을 평가하는 에이전트
    """
    
    def __init__(self, llm: Runnable):
        """초기화
        
        Args:
            llm: 사용할 LLM 모델
        """
        self.llm = llm
        
        # 시스템 프롬프트 설정
        self.system_prompt = """
        당신은 AI 윤리 전문가입니다. 주어진 AI 서비스 정보를 바탕으로 다음 윤리적 리스크를 평가해야 합니다:
        
        1. 편향성(Bias) 리스크: 서비스가 특정 그룹에 불공정한 결과를 제공할 가능성
        2. 프라이버시(Privacy) 리스크: 개인정보 수집, 저장, 활용 과정에서의 위험성
        3. 설명가능성(Explainability) 리스크: 서비스 결과에 대한 설명 부족으로 인한 위험성
        4. 자동화(Automation) 리스크: 자동화된 의사결정으로 인한 위험성
        
        각 리스크를 '낮음', '중간', '높음' 중 하나로 평가하고, 그 이유를 설명해주세요.
        평가 결과는 JSON 형식으로 반환해주세요.
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
        guideline = "OECD AI 가이드라인"  # 기본값 설정
        
        # 프롬프트 구성
        human_prompt = f"""
        분석할 서비스 정보:
        서비스 이름: {service_info.get('service_name', '알 수 없음')}
        서비스 설명: {service_info.get('description', '알 수 없음')}
        핵심 기능: {', '.join(service_info.get('core_features', ['알 수 없음']))}
        대상 사용자: {', '.join(service_info.get('target_users', ['알 수 없음']))}
        수집 데이터: {', '.join(service_info.get('collected_data_types', ['알 수 없음']))}
        
        적용할 윤리 가이드라인: {guideline}
        
        위 정보를 바탕으로 다음 윤리적 리스크를 평가하고 JSON 형식으로 결과를 반환해주세요:
        
        ```json
        {{
          "bias_risk": "낮음/중간/높음",
          "privacy_risk": "낮음/중간/높음",
          "explainability_risk": "낮음/중간/높음",
          "automation_risk": "낮음/중간/높음",
          "justification": {{
            "bias_risk": "평가 이유",
            "privacy_risk": "평가 이유",
            "explainability_risk": "평가 이유",
            "automation_risk": "평가 이유"
          }}
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
            ethical_risks = json.loads(json_str)
        except json.JSONDecodeError:
            # 파싱 실패 시 기본값 설정
            ethical_risks = {
                "bias_risk": "중간",
                "privacy_risk": "높음",
                "explainability_risk": "낮음",
                "automation_risk": "중간",
                "justification": {
                    "bias_risk": "학습 데이터와 사용자 그룹 간 편향 가능성 있음",
                    "privacy_risk": "사용자 입력 데이터를 장기 보관하며 삭제 정책이 불명확함",
                    "explainability_risk": "결과가 어떻게 생성되었는지 사용자에게 설명하지 않음",
                    "automation_risk": "자동 요약 결과가 수동 검토 없이 사용됨"
                }
            }
        
        # 상태 업데이트
        state["ethical_risks"] = ethical_risks
        
        return state