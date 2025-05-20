import os
import json
import re
from typing import Dict, Any, List

from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable

from utils.load_prompt import load_prompt_from_file

class ImprovementAgent:
    """개선안 제시 에이전트"""
    
    def __init__(self, llm: Runnable, prompt_dir: str = "./prompts"):
        self.llm = llm
        agent_name = self.__class__.__name__

        system_prompt_path = os.path.join(prompt_dir, "improvement_system.txt")
        user_prompt_template_path = os.path.join(prompt_dir, "improvement_user.txt")

        self.system_prompt = load_prompt_from_file(system_prompt_path)
        self.user_prompt_template = load_prompt_from_file(user_prompt_template_path)

        if not self.system_prompt:
            raise FileNotFoundError(f"{agent_name}: 시스템 프롬프트 파일을 로드할 수 없습니다. 경로: {system_prompt_path}")
        if not self.user_prompt_template:
            raise FileNotFoundError(f"{agent_name}: 사용자 프롬프트 템플릿 파일을 로드할 수 없습니다. 경로: {user_prompt_template_path}")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("ImprovementAgent 실행 시작...")
        ethical_risks = state.get("ethical_risks", {})
        toxic_clauses_list = state.get("toxic_clauses", []) 
        overall_clause_risk = state.get("overall_clause_risk", "평가 정보 없음")

        # 이전 단계에서 오류가 있었다면, 해당 오류를 포함하여 개선안 생성 시도 또는 오류 반환
        if state.get("error_message"):
            print(f"ImprovementAgent: 이전 단계 오류로 인해 개선안 생성 제한됨 - {state.get('error_message')}")
            # 오류가 있을 경우, 개선안 필드에 오류 정보 추가 또는 빈 값 반환
            return {"recommendations": {"error": "선행 작업 오류로 개선안 생성 불가", "details": state.get("error_message")}}


        if not ethical_risks and not toxic_clauses_list and overall_clause_risk == "평가 정보 없음":
            print("ImprovementAgent 경고: 윤리 리스크 및 독소조항 정보가 없어 개선안을 생성할 수 없습니다.")
            return {"recommendations": {"info": "분석된 리스크 또는 독소조항 정보가 없어 개선안을 생성하지 않았습니다."}}

        # 프롬프트에 전달할 독소조항 목록 문자열 생성
        toxic_clauses_str_list = []
        if isinstance(toxic_clauses_list, list) and toxic_clauses_list:
            for i, item in enumerate(toxic_clauses_list):
                if isinstance(item, dict):
                    clause_text = item.get('clause', '내용 없음')
                    reason_text = item.get('risk_reason', '이유 없음')
                    toxic_clauses_str_list.append(f"  {i+1}. 조항: \"{clause_text}\"\n     위험 이유: {reason_text}")
                else: # 예상치 못한 형식일 경우 문자열로 변환
                    toxic_clauses_str_list.append(f"  {i+1}. {str(item)}")
        
        formatted_toxic_clauses = "\n".join(toxic_clauses_str_list) if toxic_clauses_str_list else "탐지된 특정 독소조항 없음."

        # ethical_risks에서 justification이 없을 경우 대비
        justifications = ethical_risks.get("justification", {})

        human_prompt = self.user_prompt_template.format(
            bias_risk_level=ethical_risks.get("bias_risk", "정보 없음"),
            bias_risk_justification=justifications.get("bias_risk", "상세 근거 없음"),
            privacy_risk_level=ethical_risks.get("privacy_risk", "정보 없음"),
            privacy_risk_justification=justifications.get("privacy_risk", "상세 근거 없음"),
            explainability_risk_level=ethical_risks.get("explainability_risk", "정보 없음"),
            explainability_risk_justification=justifications.get("explainability_risk", "상세 근거 없음"),
            automation_risk_level=ethical_risks.get("automation_risk", "정보 없음"),
            automation_risk_justification=justifications.get("automation_risk", "상세 근거 없음"),
            overall_clause_risk_level=overall_clause_risk,
            toxic_clauses_list=formatted_toxic_clauses
        )
        
        print("ImprovementAgent: LLM 호출 중...")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = self.llm.invoke(messages)
        
        recommendations_output = {}
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                recommendations_output = json.loads(json_str)
                print("ImprovementAgent: LLM으로부터 JSON 응답 파싱 성공.")
            else: # JSON 블록이 없을 경우, 전체 내용을 파싱 시도 (덜 안정적)
                print("ImprovementAgent 경고: LLM 응답에서 명확한 JSON 블록을 찾지 못했습니다. 전체 응답 파싱 시도.")
                recommendations_output = json.loads(response.content)
        except json.JSONDecodeError as e:
            error_msg = f"ImprovementAgent 오류: LLM 응답 JSON 파싱 실패 - {e}"
            print(error_msg)
            print(f"LLM 원본 응답 (일부):\n{response.content[:500].replace(chr(0), '')}...")
            # 상태에 오류 메시지 추가 (기존 오류 메시지가 있다면 이어붙임)
            current_error = state.get("error_message", "")
            return {"error_message": (current_error + "; " if current_error else "") + error_msg, 
                    "recommendations": {"error": "개선안 JSON 파싱 실패"}}
        
        print(f"ImprovementAgent: 생성된 개선안 - {recommendations_output}")
        # recommendations 키 아래의 내용을 저장해야 함 (프롬프트 출력 형식에 따름)
        return {"recommendations": recommendations_output.get("recommendations", {})}
