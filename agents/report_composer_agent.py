import os
import json
import re
from typing import Dict, Any, List
from datetime import datetime

from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable
# Markdown 및 PDF 변환을 위한 라이브러리 임포트
from markdown import markdown # markdown2 사용 시 from markdown2 import Markdown; markdowner = Markdown()
from weasyprint import HTML, CSS

# prompts 폴더에서 프롬프트를 로드하는 함수
def load_prompt_from_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        agent_name = os.path.splitext(os.path.basename(__file__))[0]
        # 에이전트 초기화 시 예외를 발생시키도록 변경
        raise FileNotFoundError(f"오류({agent_name}): 프롬프트 파일을 찾을 수 없습니다 - {file_path}")
    except Exception as e:
        agent_name = os.path.splitext(os.path.basename(__file__))[0]
        raise IOError(f"오류({agent_name}): 프롬프트 파일 로드 중 문제 발생 - {file_path}: {e}")

class ReportComposerAgent:
    """보고서 작성 에이전트 (Markdown 및 PDF 생성)"""
    
    def __init__(self, llm: Runnable, prompt_dir: str = "./prompts", output_dir: str = "./outputs"):
        self.llm = llm
        self.output_dir = output_dir 
        agent_name = self.__class__.__name__

        system_prompt_path = os.path.join(prompt_dir, "report_composer_system.txt")
        user_prompt_template_path = os.path.join(prompt_dir, "report_composer_user.txt")

        self.system_prompt = load_prompt_from_file(system_prompt_path)
        self.user_prompt_template = load_prompt_from_file(user_prompt_template_path)

        # self.system_prompt와 self.user_prompt_template 로드 실패 시 __init__에서 예외 발생 (load_prompt_from_file 수정에 따름)

    def _format_state_for_prompt(self, state: Dict[str, Any]) -> Dict[str, str]:
        """LLM 프롬프트에 전달하기 위해 상태 정보를 문자열로 포맷합니다."""
        service_info = state.get("service_info", {"error": "서비스 정보가 없습니다.", "service_name": "UnknownService"})
        ethical_risks = state.get("ethical_risks", {"error": "윤리 리스크 정보가 없습니다."})
        
        toxic_clauses_data = {
            "toxic_clauses": state.get("toxic_clauses", []), 
            "overall_clause_risk": state.get("overall_clause_risk", "정보 없음")
        }
        if not toxic_clauses_data["toxic_clauses"]: # 빈 리스트일 경우 프롬프트 가독성을 위해
             toxic_clauses_data["toxic_clauses"] = [{"clause": "탐지된 특정 독소조항 없음", "risk_reason": "-", "potential_impact": "-", "source_document_reference": "-"}]

        recommendations = state.get("recommendations", {"error": "개선안 정보가 없습니다."})

        return {
            "service_info_json_str": json.dumps(service_info, ensure_ascii=False, indent=2),
            "ethical_risks_json_str": json.dumps(ethical_risks, ensure_ascii=False, indent=2),
            "toxic_clauses_json_str": json.dumps(toxic_clauses_data, ensure_ascii=False, indent=2),
            "recommendations_json_str": json.dumps(recommendations, ensure_ascii=False, indent=2),
        }

    def _convert_md_to_pdf(self, markdown_string: str, pdf_path: str):
        """Markdown 문자열을 PDF 파일로 변환합니다."""
        try:
            # Markdown을 HTML로 변환
            # markdown2 사용 시: html_content = markdowner.convert(markdown_string)
            html_content = markdown(markdown_string, extensions=['extra', 'nl2br', 'sane_lists', 'codehilite', 'tables', 'fenced_code'])
            
            css_style = """
                @page { size: A4; margin: 2cm; }
                body { font-family: "Noto Sans KR", sans-serif; line-height: 1.6; }
                h1, h2, h3, h4, h5, h6 { font-family: "Noto Sans KR",  serif; color: #333; }
                h1 { font-size: 24pt; margin-bottom: 0.5em; border-bottom: 2px solid #eee; padding-bottom: 0.2em;}
                h2 { font-size: 18pt; margin-top: 1.5em; margin-bottom: 0.4em; border-bottom: 1px solid #eee; padding-bottom: 0.1em;}
                h3 { font-size: 14pt; margin-top: 1.2em; margin-bottom: 0.3em; color: #444;}
                p { margin-bottom: 0.8em; }
                ul, ol { margin-bottom: 0.8em; padding-left: 1.5em; }
                li { margin-bottom: 0.2em; }
                code { font-family: "D2Coding", monospace; background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 0.9em;}
                pre > code { display: block; padding: 0.5em; overflow-x: auto; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                blockquote { border-left: 3px solid #ccc; padding-left: 1em; margin-left: 0; color: #666; }
            """
            html = HTML(string=html_content, base_url=os.path.dirname(__file__)) # base_url은 이미지 등 상대경로 위해
            css = CSS(string=css_style)
            
            html.write_pdf(pdf_path, stylesheets=[css])
            print(f"ReportComposerAgent: PDF 보고서 저장 완료 - {pdf_path}")
            return pdf_path
        except Exception as e:
            print(f"ReportComposerAgent 오류: Markdown을 PDF로 변환 중 실패 - {e}")
            print("  (HINT: WeasyPrint 및 관련 C 라이브러리(Pango, Cairo 등)가 올바르게 설치되었는지 확인하세요.)")
            print("  (HINT: 한글 폰트가 시스템에 설치되어 있고 WeasyPrint가 접근 가능한지 확인하세요.)")
            return None

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("ReportComposerAgent 실행 시작 (PDF 변환 포함)...")
        
        final_report_output = {
            "summary": "보고서 생성 중 오류 발생",
            "report_markdown": None,
            "report_pdf": None, # PDF 경로 필드 추가
            "status": "Error",
            "error_details": "알 수 없는 오류"
        }

        if state.get("error_message"):
            error_summary = f"진단 프로세스 중 오류 발생: {state.get('error_message')}"
            print(f"ReportComposerAgent: 이전 오류로 인해 간소화된 오류 보고서 생성 - {error_summary}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            service_name_val = state.get("service_info", {}).get("service_name", "unknown_service")
            service_name_prefix = service_name_val.replace(" ", "_").replace("/", "_")[:30] if service_name_val else "unknown_service"
            
            error_report_filename_md = f"error_report_{service_name_prefix}_{timestamp}.md"
            error_report_md_path = os.path.join(self.output_dir, error_report_filename_md)
            error_report_filename_pdf = f"error_report_{service_name_prefix}_{timestamp}.pdf"
            error_report_pdf_path = os.path.join(self.output_dir, error_report_filename_pdf)
            
            error_report_content = f"# AI 윤리 리스크 진단 오류 보고서\n\n## 오류 요약\n{error_summary}\n\n## 현재까지 수집된 정보 (일부)\n"
            if state.get("service_info"): error_report_content += f"\n### 서비스 정보\n```json\n{json.dumps(state.get('service_info'), ensure_ascii=False, indent=2)}\n```\n"
            # ... (다른 정보들도 필요시 추가) ...
            
            os.makedirs(self.output_dir, exist_ok=True)
            md_path_saved = None
            pdf_path_saved = None
            try:
                with open(error_report_md_path, "w", encoding="utf-8") as f:
                    f.write(error_report_content)
                print(f"ReportComposerAgent: 오류 보고서(MD) 저장 완료 - {error_report_md_path}")
                md_path_saved = error_report_md_path
                # 오류 보고서도 PDF로 변환 시도
                pdf_path_saved = self._convert_md_to_pdf(error_report_content, error_report_pdf_path)
            except Exception as e_save:
                 print(f"ReportComposerAgent 오류: 오류 보고서 저장/변환 실패 - {e_save}")

            final_report_output["summary"] = error_summary
            final_report_output["report_markdown"] = md_path_saved
            final_report_output["report_pdf"] = pdf_path_saved
            final_report_output["error_details"] = state.get("error_message")
            return {"final_report": final_report_output}

        prompt_inputs = self._format_state_for_prompt(state)
        human_prompt = self.user_prompt_template.format(**prompt_inputs)
        
        print("ReportComposerAgent: LLM 호출 중 (최종 보고서 생성)...")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = self.llm.invoke(messages)
        
        report_content_markdown = response.content.replace(chr(0), '')
        
        summary = "요약 정보를 찾을 수 없습니다."
        summary_match = re.search(r'SUMMARY\s*:\s*(.*?)(?=\n\n##|\n\n#|\Z)', report_content_markdown, re.DOTALL | re.IGNORECASE)
        if summary_match:
            summary = summary_match.group(1).strip()
        else:
            summary_lines = report_content_markdown.split('\n')
            summary_candidate = "\n".join(line for line in summary_lines[:10] if line.strip() and not line.strip().startswith('#')) 
            summary = summary_candidate if summary_candidate else "보고서 요약 자동 추출 실패."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        service_name_val = state.get("service_info", {}).get("service_name", "unknown_service")
        service_name_prefix = service_name_val.replace(" ", "_").replace("/", "_")[:30] if service_name_val else "unknown_service"
        
        base_filename = f"ethics_report_{service_name_prefix}_{timestamp}"
        report_markdown_filename = f"{base_filename}.md"
        report_markdown_path = os.path.join(self.output_dir, report_markdown_filename)
        report_pdf_filename = f"{base_filename}.pdf"
        report_pdf_path = os.path.join(self.output_dir, report_pdf_filename)
        
        os.makedirs(self.output_dir, exist_ok=True)
        md_saved_path = None
        pdf_saved_path = None
        try:
            with open(report_markdown_path, "w", encoding="utf-8") as f:
                f.write(report_content_markdown)
            md_saved_path = report_markdown_path
            print(f"ReportComposerAgent: Markdown 보고서 저장 완료 - {md_saved_path}")

            # Markdown을 PDF로 변환
            pdf_saved_path = self._convert_md_to_pdf(report_content_markdown, report_pdf_path)
            
        except Exception as e: # 파일 저장 또는 PDF 변환 중 오류
            print(f"ReportComposerAgent 오류: 보고서 저장 또는 PDF 변환 실패 - {e}")
            final_report_output["summary"] = summary # 요약은 성공했을 수 있음
            final_report_output["report_markdown"] = md_saved_path # md 저장은 성공했을 수 있음
            final_report_output["status"] = "Partial Success (Save/Convert Failed)"
            final_report_output["error_details"] = f"보고서 저장/PDF 변환 실패: {str(e)}"
            return {"final_report": final_report_output, "error_message": state.get("error_message","") + f"; Report File Save/Convert Failed: {str(e)}"}
        
        final_report_output["summary"] = summary
        final_report_output["report_markdown"] = md_saved_path
        final_report_output["report_pdf"] = pdf_saved_path
        final_report_output["status"] = "Success" if pdf_saved_path else "Partial Success (PDF Convert Failed)"
        final_report_output.pop("error_details", None) # 성공 시에는 error_details 제거

        return {"final_report": final_report_output}
