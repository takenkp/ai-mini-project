from typing import Dict, Any, List
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable

class ServiceAnalysisAgent:
    """서비스 분석 에이전트
    
    서비스의 기능, 대상 사용자, 수집 데이터 등을 분석하는 에이전트
    """
    
    def __init__(self, llm: Runnable):
        """초기화
        
        Args:
            llm: 사용할 LLM 모델
        """
        self.llm = llm
        
        # 시스템 프롬프트 설정
        self.system_prompt = """
        당신은 AI 서비스 분석 전문가입니다. 주어진 서비스 URL과 문서를 분석하여 다음 정보를 추출해야 합니다:
        
        1. 서비스 이름
        2. 서비스 설명
        3. 핵심 기능 목록
        4. 대상 사용자 그룹
        5. 수집하는 데이터 유형
        
        분석 결과는 JSON 형식으로 반환해주세요.
        """
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 실행
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        # 입력 데이터 추출
        service_url = state.get("service_url", "")
        documents = state.get("documents", [])
        
        # 문서 내용 로드 (실제 구현에서는 PDF 파서 등 사용)
        document_contents = "문서 내용을 여기에 로드합니다. (실제 구현 시 PyMuPDF 등 사용)"
        
        # 프롬프트 구성
        human_prompt = f"""
        분석할 서비스 URL: {service_url}
        
        문서 내용:
        {document_contents}
        
        위 정보를 바탕으로 서비스를 분석하고 다음 JSON 형식으로 결과를 반환해주세요:
        
        ```json
        {{
          "service_name": "서비스 이름",
          "description": "서비스 설명",
          "core_features": ["기능1", "기능2", ...],
          "target_users": ["사용자 그룹1", "사용자 그룹2", ...],
          "collected_data_types": ["데이터 유형1", "데이터 유형2", ...]
        }}
        ```
        """
        
        # LLM 호출
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = self.llm.invoke(messages)
        
        # 응답에서 JSON 추출 (실제 구현에서는 더 견고한 파싱 필요)
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
            service_info = json.loads(json_str)
        except json.JSONDecodeError:
            # 파싱 실패 시 기본값 설정
            service_info = {
                "service_name": "Daglo AI Guide",
                "description": "사용자가 입력한 문서를 요약해주는 대화형 AI 가이드",
                "core_features": ["문서 요약", "인터랙티브 가이드", "챗 기반 추천"],
                "target_users": ["기업 사용자", "일반 사용자"],
                "collected_data_types": ["텍스트 입력", "클릭 로그"]
            }
        
        # 상태 업데이트
        state["service_info"] = service_info
        
        return state