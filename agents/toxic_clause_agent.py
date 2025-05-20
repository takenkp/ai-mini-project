import os
import json
import re
from typing import Dict, Any, List

from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable
from langchain.retrievers.ensemble import EnsembleRetriever

from utils.load_prompt import load_prompt_from_file

class ToxicClauseAgent:
    """독소조항 탐지 에이전트 (RAG 적용, terms/privacy 텍스트 직접 입력 받지 않음)"""
    
    def __init__(self, llm: Runnable, retriever: EnsembleRetriever | None, prompt_dir: str = "./prompts"):
        self.llm = llm
        self.retriever = retriever
        agent_name = self.__class__.__name__

        system_prompt_path = os.path.join(prompt_dir, "toxic_clause_system.txt")
        user_prompt_template_path = os.path.join(prompt_dir, "toxic_clause_user.txt")

        self.system_prompt = load_prompt_from_file(system_prompt_path)
        self.user_prompt_template = load_prompt_from_file(user_prompt_template_path)

        if not self.system_prompt:
            raise FileNotFoundError(f"{agent_name}: 시스템 프롬프트 파일을 로드할 수 없습니다. 경로: {system_prompt_path}")
        if not self.user_prompt_template:
            raise FileNotFoundError(f"{agent_name}: 사용자 프롬프트 템플릿 파일을 로드할 수 없습니다. 경로: {user_prompt_template_path}")

    def _get_rag_context_for_legal_analysis(self, service_info: Dict[str, Any], documents_to_consider: List[str]) -> str:
        """서비스의 약관, 개인정보처리방침 등 법적 문서 관련 내용을 RAG로 검색합니다."""
        if not self.retriever:
            return "Retriever가 제공되지 않아 약관/개인정보 관련 컨텍스트를 가져올 수 없습니다.\n"
        
        service_name = service_info.get("service_name", "해당 서비스")
        query_keywords = ["이용약관", "서비스 약관", "개인정보 처리방침", "개인정보 보호정책", "데이터 사용 정책", "사용자 권리", "책임 제한", "면책 조항", "계약 변경", "서비스 중단"]
        query = f"'{service_name}' 서비스의 공식 문서 또는 웹사이트 내용 중 다음 키워드와 관련된 법적 조항, 정책, 또는 사용자에게 영향을 미칠 수 있는 중요한 고지 사항을 찾아주세요: {', '.join(query_keywords)}."
        
        if documents_to_consider: # documents는 PDF 파일 경로 리스트
            doc_names = ", ".join([os.path.basename(doc_path) for doc_path in documents_to_consider])
            query += f" (주요 참고 문서: {doc_names})"
            
        print(f"ToxicClauseAgent: RAG 쿼리 - \"{query[:250]}...\"") # 쿼리 길이 제한 출력
        
        try:
            if hasattr(self.retriever, 'invoke'):
                relevant_docs = self.retriever.invoke(query)
            elif hasattr(self.retriever, 'get_relevant_documents'):
                relevant_docs = self.retriever.get_relevant_documents(query)
            else:
                return "Retriever에 적절한 검색 메소드가 없습니다.\n"
        except Exception as e:
            return f"RAG 컨텍스트 검색 중 오류 발생 ({e})\n"
            
        context_str = "## 서비스 약관 및 개인정보 처리방침 관련 문서 컨텍스트 (RAG 결과):\n"
        if not relevant_docs:
            context_str += "  관련된 법적 내용을 문서에서 찾을 수 없습니다.\n"
            return context_str

        for i, doc in enumerate(relevant_docs):
            source_file = doc.metadata.get('source_file', doc.metadata.get('source', 'N/A'))
            page_num = doc.metadata.get('page', 'N/A')
            section_title = doc.metadata.get('section_title', 'N/A') # indexer.py에서 추가한 메타데이터
            context_str += (
                f"  --- 컨텍스트 {i+1} (출처: {source_file}, 페이지: {page_num}, 섹션: {section_title}) ---\n"
                f"  {doc.page_content[:250].replace(chr(0), '')}...\n" # 컨텍스트 길이 약간 늘림, NULL 바이트 제거
            )
            print(context_str) # 디버깅용 출력 (항목별 컨텍스트)
        return context_str + "\n"

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("ToxicClauseAgent 실행 시작...")
        # terms_text와 privacy_policy_text를 직접 받는 대신 RAG로 가져옴
        service_info = state.get("service_info", {}) 
        documents = state.get("documents", []) # 분석 대상 PDF 문서 경로

        if not service_info and not documents:
            print("ToxicClauseAgent 경고: 서비스 정보와 문서 경로가 모두 없어 분석이 제한됩니다.")
            # 이 경우 RAG도 불가능하므로, 기본 오류 반환
            return {"error_message": "독소조항 분석을 위한 정보 부족 (서비스 정보 및 문서 없음)", "toxic_clauses": [], "overall_clause_risk": "평가 불가"}

        # RAG 컨텍스트 생성
        rag_context = self._get_rag_context_for_legal_analysis(service_info, documents)
        
        # 사용자 프롬프트는 이제 terms_text, privacy_policy_text 대신 service_info의 일부와 rag_context를 받음
        human_prompt = self.user_prompt_template.format(
            service_name=service_info.get('service_name', '알 수 없음'),
            description=service_info.get('description', '알 수 없음'), # 필요시 프롬프트 템플릿에 추가
            rag_context_toxic_clause=rag_context
        )
        
        print("ToxicClauseAgent: LLM 호출 중...")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = self.llm.invoke(messages)
        
        toxic_clause_output = {}
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                toxic_clause_output = json.loads(json_str)
                print("ToxicClauseAgent: LLM으로부터 JSON 응답 파싱 성공.")
            else:
                print("ToxicClauseAgent 경고: LLM 응답에서 명확한 JSON 블록을 찾지 못했습니다. 전체 응답 파싱 시도.")
                toxic_clause_output = json.loads(response.content)
        except json.JSONDecodeError as e:
            print(f"ToxicClauseAgent 오류: LLM 응답 JSON 파싱 실패 - {e}")
            print(f"LLM 원본 응답 (일부):\n{response.content[:500].replace(chr(0), '')}...")
            toxic_clause_output = {"error_message": "JSON 파싱 실패", "toxic_clauses": [], "overall_clause_risk": "평가 불가"}
        
        print(f"ToxicClauseAgent: 탐지된 독소 조항 정보 - {toxic_clause_output}")
        # 에러 메시지가 있다면 상태에 포함
        if "error_message" in toxic_clause_output:
             return {
                "toxic_clauses": toxic_clause_output.get("toxic_clauses", []),
                "overall_clause_risk": toxic_clause_output.get("overall_clause_risk", "평가 불가"),
                "error_message": state.get("error_message", "") + "; " + toxic_clause_output["error_message"] # 기존 오류에 추가
            }
        return {
            "toxic_clauses": toxic_clause_output.get("toxic_clauses", []),
            "overall_clause_risk": toxic_clause_output.get("overall_clause_risk", "평가 불가")
        }
