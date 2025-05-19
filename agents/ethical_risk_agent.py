import os
import json
import re
from typing import Dict, Any, List

from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable
from langchain.retrievers.ensemble import EnsembleRetriever # 타입 힌트용

# prompts 폴더에서 프롬프트를 로드하는 함수
def load_prompt_from_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        agent_name = os.path.splitext(os.path.basename(__file__))[0] 
        print(f"오류({agent_name}): 프롬프트 파일을 찾을 수 없습니다 - {file_path}")
        return "" 
    except Exception as e:
        agent_name = os.path.splitext(os.path.basename(__file__))[0]
        print(f"오류({agent_name}): 프롬프트 파일 로드 중 문제 발생 - {file_path}: {e}")
        return ""

class EthicalRiskAgent:
    """윤리적 리스크 평가 에이전트 (RAG 및 특정 가이드라인 참조 적용)"""
    
    def __init__(self, llm: Runnable, retriever: EnsembleRetriever | None, 
                 guideline_doc_keyword: str = "OECD", # RAG 쿼리 시 참조할 가이드라인 문서 키워드
                 prompt_dir: str = "./prompts"):
        self.llm = llm
        self.retriever = retriever
        self.guideline_doc_keyword = guideline_doc_keyword # 예: "OECD", "AI 윤리 가이드라인" 등
        agent_name = self.__class__.__name__

        system_prompt_path = os.path.join(prompt_dir, "ethical_risk_system.txt")
        user_prompt_template_path = os.path.join(prompt_dir, "ethical_risk_user.txt")

        self.system_prompt = load_prompt_from_file(system_prompt_path)
        self.user_prompt_template = load_prompt_from_file(user_prompt_template_path)

        if not self.system_prompt:
            raise FileNotFoundError(f"{agent_name}: 시스템 프롬프트 파일을 로드할 수 없습니다. 경로: {system_prompt_path}")
        if not self.user_prompt_template:
            raise FileNotFoundError(f"{agent_name}: 사용자 프롬프트 템플릿 파일을 로드할 수 없습니다. 경로: {user_prompt_template_path}")

        self.ethical_risk_items_for_rag = {
            "bias_risk": "서비스의 잠재적인 편향성(Bias) 리스크",
            "privacy_risk": "서비스의 개인정보보호(Privacy) 관련 리스크",
            "explainability_risk": "서비스 결과의 설명가능성(Explainability) 관련 리스크",
            "automation_risk": "서비스의 자동화된 의사결정(Automation) 관련 리스크"
        }

    def _get_rag_context_for_item(self, item_description: str, service_info: Dict[str, Any], documents_to_consider: List[str]) -> str:
        if not self.retriever:
            return f"  - '{item_description}' 관련 컨텍스트: Retriever가 제공되지 않았습니다.\n"

        service_name = service_info.get("service_name", "해당 AI 서비스")
        # RAG 쿼리에 윤리 가이드라인 키워드 포함
        query = f"'{service_name}'의 '{item_description}'에 대해, 특히 '{self.guideline_doc_keyword}' 가이드라인을 참조하여 관련된 정책, 기능, 데이터 처리 방식, 또는 잠재적 문제점을 관련 문서에서 찾아주세요."
        
        # RAG 시스템이 이 문서들 내에서 쿼리와 관련된 내용을 찾음
        if documents_to_consider:
            doc_names_preview = [os.path.basename(doc_path) for doc_path in documents_to_consider[:2]] # 너무 많으면 일부만 표시
            query += f" (주요 참고 문서 예시: {', '.join(doc_names_preview)}{' 등' if len(documents_to_consider) > 2 else ''})"
        
        print(f"EthicalRiskAgent: RAG 쿼리 (항목: {item_description}) - \"{query[:150]}...\"")
        
        try:
            if hasattr(self.retriever, 'invoke'):
                relevant_docs = self.retriever.invoke(query)
            elif hasattr(self.retriever, 'get_relevant_documents'):
                relevant_docs = self.retriever.get_relevant_documents(query)
            else:
                return f"  - '{item_description}' 관련 컨텍스트: Retriever에 적절한 검색 메소드가 없습니다.\n"
        except Exception as e:
            return f"  - '{item_description}' 관련 컨텍스트: RAG 검색 중 오류 발생 ({e})\n"
            
        context_str = f"  - '{item_description}' (특히 '{self.guideline_doc_keyword}' 가이드라인 참조) 관련 RAG 검색 결과:\n"
        if not relevant_docs:
            context_str += "    관련 문서를 찾을 수 없습니다.\n"
            return context_str

        for i, doc in enumerate(relevant_docs):
            source_file = doc.metadata.get('source_file', doc.metadata.get('source', 'N/A'))
            page_num = doc.metadata.get('page', 'N/A')
            section_title = doc.metadata.get('section_title', 'N/A')
            context_str += (
                f"    --- 컨텍스트 {i+1} (출처: {source_file}, 페이지: {page_num}, 섹션: {section_title}) ---\n"
                f"    {doc.page_content[:300].replace(chr(0), '')}...\n"
            )
        return context_str + "\n"

    def _get_comprehensive_rag_context(self, service_info: Dict[str, Any], documents_to_consider: List[str]) -> str:
        comprehensive_context = "## 각 윤리 리스크 항목별 관련 문서 컨텍스트 (윤리 가이드라인 포함):\n"
        if not self.retriever:
            comprehensive_context += "Retriever가 제공되지 않아 RAG를 수행할 수 없습니다.\n"
            return comprehensive_context

        for item_key, item_desc_for_query in self.ethical_risk_items_for_rag.items():
            context_for_item = self._get_rag_context_for_item(item_desc_for_query, service_info, documents_to_consider)
            comprehensive_context += context_for_item
        
        return comprehensive_context
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("EthicalRiskAgent 실행 시작 (가이드라인 참조 강화)...")
        service_info = state.get("service_info", {})
        documents = state.get("documents", []) # 여기에는 서비스 문서 + 윤리 가이드라인 문서 경로가 포함되어야 함

        if not service_info:
            print("EthicalRiskAgent 경고: 서비스 정보(service_info)가 없습니다.")
            return {"ethical_risks": {"error": "서비스 정보 부족"}, "ethical_risk_done": True} # 실패해도 done 플래그 설정

        rag_context = self._get_comprehensive_rag_context(service_info, documents)
        
        human_prompt = self.user_prompt_template.format(
            service_name=service_info.get('service_name', '알 수 없음'),
            description=service_info.get('description', '알 수 없음'),
            core_features=", ".join(service_info.get('core_features', ['정보 없음'])),
            target_users=", ".join(service_info.get('target_users', ['정보 없음'])),
            collected_data_types=", ".join(service_info.get('collected_data_types', ['정보 없음'])),
            rag_context_ethical_risk=rag_context # RAG 결과 주입
        )
        
        print("EthicalRiskAgent: LLM 호출 중...")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = self.llm.invoke(messages)
        
        ethical_risks_output = {}
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                ethical_risks_output = json.loads(json_str)
                print("EthicalRiskAgent: LLM으로부터 JSON 응답 파싱 성공.")
            else:
                print("EthicalRiskAgent 경고: LLM 응답에서 명확한 JSON 블록을 찾지 못했습니다. 전체 응답 파싱 시도.")
                ethical_risks_output = json.loads(response.content) 
        except json.JSONDecodeError as e:
            error_msg = f"EthicalRiskAgent 오류: LLM 응답 JSON 파싱 실패 - {e}"
            print(error_msg)
            print(f"LLM 원본 응답 (일부):\n{response.content[:500].replace(chr(0), '')}...")
            return {"error_message": state.get("error_message","") + "; " + error_msg, "ethical_risks": {"error": "JSON 파싱 실패"}, "ethical_risk_done": True}
        
        print(f"EthicalRiskAgent: 평가된 윤리 리스크 - {ethical_risks_output}")
        return {"ethical_risks": ethical_risks_output, "ethical_risk_done": True}
