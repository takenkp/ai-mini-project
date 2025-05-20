import os
import json
import re
from typing import Dict, Any, List

from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable
from langchain.retrievers.ensemble import EnsembleRetriever # 타입 힌트용

from utils.load_prompt import load_prompt_from_file

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
        """
        특정 평가 항목(item_description)에 대해 다양한 윤리적 측면을 고려하여 RAG로 관련 컨텍스트를 검색합니다.
        각 측면별로 RAG 쿼리를 생성하고 결과를 취합합니다.
        """
        if not self.retriever:
            return f"  - '{item_description}' 관련 컨텍스트: Retriever가 제공되지 않았습니다.\n"

        service_name = service_info.get("service_name", "해당 AI 서비스")

        # 각 항목에 대해 검색할 윤리적 측면 또는 세부 키워드 리스트
        # 이 키워드들은 self.guideline_doc_keyword 와 함께 사용되어 검색 쿼리를 구체화합니다.
        ethical_aspect_keywords = [
            "데이터 수집 및 처리의 적절성",
            "개인정보보호 및 프라이버시 침해 가능성",
            "알고리즘 편향성 및 공정성 문제",
            "투명성 및 설명가능성 확보 방안",
            "결과의 신뢰성 및 안전성",
            "오용 및 악용 방지 대책",
            "사회적 차별 또는 불평등 야기 가능성",
            "사용자 통제권 및 자율성 보장",
            # 필요에 따라 서비스 특성 및 guideline_doc_keyword에 맞춰 키워드 추가/수정
        ]

        # 문서 경로가 있다면 쿼리에 포함시킬 문서명 문자열 생성
        doc_names_suffix = ""
        if documents_to_consider:
            # 문서가 너무 많으면 일부만 표시 (예: 처음 3개)
            doc_names_preview = [os.path.basename(doc_path) for doc_path in documents_to_consider[:3]]
            suffix_etc = " 등" if len(documents_to_consider) > 3 else ""
            doc_names_suffix = f" (주요 참고 문서 예시: {', '.join(doc_names_preview)}{suffix_etc})"

        # 최종 컨텍스트 문자열을 빌드하기 위한 리스트
        item_all_contexts_parts = [f"\n## '{item_description}' 항목 관련 윤리적 분석 컨텍스트 (RAG 결과):\n"]
        item_all_contexts_parts.append(f"   (주요 참조 가이드라인 키워드: '{self.guideline_doc_keyword}')\n")
        
        found_any_context_for_item = False

        for aspect_keyword in ethical_aspect_keywords:
            # 각 윤리적 측면에 대한 특정 RAG 쿼리 생성
            query = (
                f"'{service_name}' 서비스의 '{item_description}' 기능/항목과 관련하여, "
                f"'{aspect_keyword}' 측면에 대해 '{self.guideline_doc_keyword}' 가이드라인을 참조했을 때, "
                f"관련된 정책, 기술적 구현, 데이터 처리 방식, 잠재적 위험 또는 완화 조치 등을 설명하는 내용을 찾아주세요."
                f"{doc_names_suffix}"
            )

            print(f"EthicalRiskAgent: RAG 쿼리 (항목: {item_description}, 측면: {aspect_keyword}) - \"{query[:180]}...\"")

            relevant_docs_for_aspect = []
            try:
                if hasattr(self.retriever, 'invoke'):
                    relevant_docs_for_aspect = self.retriever.invoke(query)
                elif hasattr(self.retriever, 'get_relevant_documents'):
                    relevant_docs_for_aspect = self.retriever.get_relevant_documents(query)
                else:
                    item_all_contexts_parts.append(f"\n### '{item_description}'의 '{aspect_keyword}' 측면:\n")
                    item_all_contexts_parts.append(f"  - Retriever에 적절한 검색 메소드가 없어 컨텍스트를 가져올 수 없습니다.\n")
                    continue # 다음 측면 키워드로
            except Exception as e:
                item_all_contexts_parts.append(f"\n### '{item_description}'의 '{aspect_keyword}' 측면:\n")
                item_all_contexts_parts.append(f"  - RAG 컨텍스트 검색 중 오류 발생 ({e})\n")
                continue # 다음 측면 키워드로

            if relevant_docs_for_aspect:
                found_any_context_for_item = True
                item_all_contexts_parts.append(f"\n### '{item_description}'의 '{aspect_keyword}' 측면:\n")
                for i, doc in enumerate(relevant_docs_for_aspect):
                    source_file = doc.metadata.get('source_file', doc.metadata.get('source', 'N/A'))
                    page_num = doc.metadata.get('page', 'N/A')
                    section_title = doc.metadata.get('section_title', 'N/A') # indexer.py 등에서 추가한 메타데이터 가정
                    content_preview = doc.page_content.replace(chr(0), '').strip()[:300] # NULL 바이트 제거 및 미리보기 길이 조정

                    item_all_contexts_parts.append(
                        f"  --- 컨텍스트 {i+1} (출처: {source_file}, 페이지: {page_num}, 섹션: {section_title}) ---\n"
                        f"  {content_preview}...\n"
                    )
                    # 상세 디버깅이 필요할 때 아래 프린트문 주석 해제
                    # print(f"Debug: Item='{item_description}', Aspect='{aspect_keyword}', Source='{source_file}', Page='{page_num}', Content='{content_preview[:50]}...'")
            else:
                item_all_contexts_parts.append(f"\n### '{item_description}'의 '{aspect_keyword}' 측면:\n")
                item_all_contexts_parts.append(f"  - 이 측면에 대해 '{self.guideline_doc_keyword}' 가이드라인을 참조하여 검색된 관련 내용을 문서에서 찾을 수 없습니다.\n")

        if not found_any_context_for_item and len(item_all_contexts_parts) <= 2 : # 헤더와 가이드라인 키워드 안내만 있는 경우
            item_all_contexts_parts.append(f"  '{item_description}' 항목에 대해 모든 윤리적 측면에서 관련된 내용을 찾을 수 없었습니다.\n")
            
        return "".join(item_all_contexts_parts) + "\n"
        
    def _get_comprehensive_rag_context(self, service_info: Dict[str, Any], documents_to_consider: List[str]) -> str:
        comprehensive_context = "## 각 윤리 리스크 항목별 관련 문서 컨텍스트 (윤리 가이드라인 포함):\n"
        if not self.retriever:
            comprehensive_context += "Retriever가 제공되지 않아 RAG를 수행할 수 없습니다.\n"
            return comprehensive_context

        for item_desc_for_query in self.ethical_risk_items_for_rag.items():
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
