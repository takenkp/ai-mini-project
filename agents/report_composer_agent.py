import os
import json
import re
from typing import Dict, Any, List
from datetime import datetime

from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable
# import textwrap # 사용자 프롬프트가 매우 길 경우 사용 가능

# prompts 폴더에서 프롬프트를 로드하는 함수
def load_prompt_from_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        agent_name = os.path.splitext(os.path.basename(__file__))[0]
        print(f"경고({agent_name}): 프롬프트 파일을 찾을 수 없습니다 - {file_path}")
        return ""
    except Exception as e:
        agent_name = os.path.splitext(os.path.basename(__file__))[0]
        print(f"오류({agent_name}): 프롬프트 파일 로드 중 문제 발생 - {file_path}: {e}")
        return ""

class ReportComposerAgent:
    """보고서 작성 에이전트"""
    
    def __init__(self, llm: Runnable, prompt_dir: str = "./prompts", output_dir: str = "./outputs"):
        self.llm = llm
        self.output_dir = output_dir 
        agent_name = self.__class__.__name__

        system_prompt_path = os.path.join(prompt_dir, "report_composer_system.txt")
        user_prompt_template_path = os.path.join(prompt_dir, "report_composer_user.txt")

        self.system_prompt = load_prompt_from_file(system_prompt_path)
        self.user_prompt_template = load_prompt_from_file(user_prompt_template_path)

        if not self.system_prompt:
            raise FileNotFoundError(f"{agent_name}: 시스템 프롬프트 파일을 로드할 수 없습니다. 경로: {system_prompt_path}")
        if not self.user_prompt_template:
            raise FileNotFoundError(f"{agent_name}: 사용자 프롬프트 템플릿 파일을 로드할 수 없습니다. 경로: {user_prompt_template_path}")

    def _format_state_for_prompt(self, state: Dict[str, Any]) -> Dict[str, str]:
        """LLM 프롬프트에 전달하기 위해 상태 정보를 문자열로 포맷합니다."""
        service_info = state.get("service_info", {"error": "서비스 정보가 없습니다.", "service_name": "UnknownService"})
        ethical_risks = state.get("ethical_risks", {"error": "윤리 리스크 정보가 없습니다."})
        
        toxic_clauses_data = {
            "toxic_clauses": state.get("toxic_clauses", []), # 기본값 빈 리스트
            "overall_clause_risk": state.get("overall_clause_risk", "정보 없음")
        }
        # toxic_clauses가 비어있으면 "탐지된 독소조항 없음"으로 표시되도록 프롬프트에서 처리하거나 여기서 가공
        if not toxic_clauses_data["toxic_clauses"]:
             toxic_clauses_data["toxic_clauses"] = [{"clause": "탐지된 특정 독소조항 없음", "risk_reason": "-"}]


        recommendations = state.get("recommendations", {"error": "개선안 정보가 없습니다."})

        return {
            "service_info_json_str": json.dumps(service_info, ensure_ascii=False, indent=2),
            "ethical_risks_json_str": json.dumps(ethical_risks, ensure_ascii=False, indent=2),
            "toxic_clauses_json_str": json.dumps(toxic_clauses_data, ensure_ascii=False, indent=2),
            "recommendations_json_str": json.dumps(recommendations, ensure_ascii=False, indent=2),
        }

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("ReportComposerAgent 실행 시작...")
        
        # 이전 단계에서 심각한 오류가 있었는지 확인
        # error_message는 주로 파이프라인 레벨의 오류나 join 타임아웃 등에 사용됨
        # 각 에이전트의 결과 내에도 error 필드가 있을 수 있음 (예: ethical_risks.get("error"))
        if state.get("error_message"):
            error_summary = f"진단 프로세스 중 오류 발생: {state.get('error_message')}"
            print(f"ReportComposerAgent: 이전 오류로 인해 간소화된 오류 보고서 생성 - {error_summary}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 서비스 이름이 service_info에 없을 수 있으므로 기본값 사용
            service_name_val = state.get("service_info", {}).get("service_name", "unknown_service")
            service_name_prefix = service_name_val.replace(" ", "_")[:30] if service_name_val else "unknown_service"
            
            error_report_filename = f"error_report_{service_name_prefix}_{timestamp}.md"
            error_report_md_path = os.path.join(self.output_dir, error_report_filename)
            
            error_report_content = f"# AI 윤리 리스크 진단 오류 보고서\n\n## 오류 요약\n{error_summary}\n\n## 현재까지 수집된 정보 (일부)\n"
            # 상태에 있는 주요 정보들을 최대한 포함
            if state.get("service_info"): error_report_content += f"\n### 서비스 정보\n```json\n{json.dumps(state.get('service_info'), ensure_ascii=False, indent=2)}\n```\n"
            if state.get("ethical_risks"): error_report_content += f"\n### 윤리 리스크\n```json\n{json.dumps(state.get('ethical_risks'), ensure_ascii=False, indent=2)}\n```\n"
            # toxic_clauses와 overall_clause_risk를 함께 표시
            toxic_data_for_error_report = {
                "toxic_clauses": state.get("toxic_clauses", []),
                "overall_clause_risk": state.get("overall_clause_risk", "정보 없음")
            }
            error_report_content += f"\n### 독소 조항\n```json\n{json.dumps(toxic_data_for_error_report, ensure_ascii=False, indent=2)}\n```\n"
            if state.get("recommendations"): error_report_content += f"\n### 개선안\n```json\n{json.dumps(state.get('recommendations'), ensure_ascii=False, indent=2)}\n```\n"
            
            os.makedirs(self.output_dir, exist_ok=True)
            try:
                with open(error_report_md_path, "w", encoding="utf-8") as f:
                    f.write(error_report_content)
                print(f"ReportComposerAgent: 오류 보고서 저장 완료 - {error_report_md_path}")
            except Exception as e_save:
                 print(f"ReportComposerAgent 오류: 오류 보고서 저장 실패 - {e_save}")
                 error_report_md_path = None # 저장 실패 시 경로 없음

            return {
                "final_report": {
                    "summary": error_summary,
                    "report_markdown": error_report_md_path,
                    "status": "Error",
                    "error_details": state.get("error_message")
                }
            }

        prompt_inputs = self._format_state_for_prompt(state)
        human_prompt = self.user_prompt_template.format(**prompt_inputs)
        
        print("ReportComposerAgent: LLM 호출 중 (최종 보고서 생성)...")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = self.llm.invoke(messages)
        
        report_content_markdown = response.content.replace(chr(0), '') # NULL 바이트 제거
        
        summary = "요약 정보를 찾을 수 없습니다."
        # SUMMARY 태그가 대소문자 구분 없이, 콜론 뒤 공백 유무에 관계없이, 줄바꿈 전까지 매칭하도록 수정
        summary_match = re.search(r'SUMMARY\s*:\s*(.*?)(?=\n\n##|\n\n#|\Z)', report_content_markdown, re.DOTALL | re.IGNORECASE)
        if summary_match:
            summary = summary_match.group(1).strip()
        else:
            summary_lines = report_content_markdown.split('\n')
            summary_candidate = "\n".join(line for line in summary_lines[:10] if line.strip() and not line.strip().startswith('#'))
            summary = summary_candidate if summary_candidate else "보고서 요약 자동 추출 실패."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        service_name_val = state.get("service_info", {}).get("service_name", "unknown_service")
        service_name_prefix = service_name_val.replace(" ", "_").replace("/", "_")[:30] if service_name_val else "unknown_service" # 파일명에 부적절한 문자 제거
        
        report_markdown_filename = f"ethics_report_{service_name_prefix}_{timestamp}.md"
        report_markdown_path = os.path.join(self.output_dir, report_markdown_filename)
        
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            with open(report_markdown_path, "w", encoding="utf-8") as f:
                f.write(report_content_markdown)
            print(f"ReportComposerAgent: Markdown 보고서 저장 완료 - {report_markdown_path}")
        except Exception as e:
            print(f"ReportComposerAgent 오류: Markdown 보고서 저장 실패 - {e}")
            return {
                "final_report": {
                    "summary": "보고서 생성은 성공했으나 파일 저장 실패",
                    "report_markdown": None, # 저장 실패
                    "status": "Partial Success (Save Failed)",
                    "error_details": f"Markdown 파일 저장 실패: {e}"
                },
                # 이전 단계의 오류가 없었으므로 error_message는 설정하지 않거나, 파일 저장 오류만 기록
                 "error_message": state.get("error_message","") + f"; Report File Save Failed: {str(e)}"
            }
        
        return {
            "final_report": {
                "summary": summary,
                "report_markdown": report_markdown_path,
                "status": "Success"
            }
        }
