import os
import json
import re
from typing import Dict, Any, List

from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable
from langchain.retrievers.ensemble import EnsembleRetriever # EnsembleRetriever 타입 힌트를 위해 직접 임포트

def load_prompt_from_file(file_path: str) -> str:
    """지정된 경로의 파일에서 프롬프트 내용을 읽어 반환합니다."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"오류: 프롬프트 파일을 찾을 수 없습니다 - {file_path}")
        return ""
    except Exception as e:
        print(f"오류: 프롬프트 파일 로드 중 문제 발생 - {file_path}: {e}")
        return ""

class ServiceAnalysisAgent:
    """서비스 분석 에이전트
    
    서비스의 기능, 대상 사용자, 수집 데이터 등을 RAG를 활용하여 분석하는 에이전트
    각 주요 정보 항목에 대해 개별 RAG 쿼리를 수행하여 컨텍스트의 관련성을 높입니다.
    """
    
    def __init__(self, llm: Runnable, retriever: EnsembleRetriever, prompt_dir: str = "./prompts"):
        """초기화
        
        Args:
            llm: 사용할 LLM 모델
            retriever: 미리 초기화된 EnsembleRetriever 인스턴스
            prompt_dir: 프롬프트 파일이 있는 디렉토리 경로
        """
        self.llm = llm
        self.retriever = retriever
        
        if self.retriever is None:
            print("ServiceAnalysisAgent 경고: 유효한 Retriever가 주입되지 않았습니다. RAG 기능이 제한될 수 있습니다.")

        system_prompt_path = os.path.join(prompt_dir, "service_analysis_system.txt")
        user_prompt_template_path = os.path.join(prompt_dir, "service_analysis_user.txt")

        self.system_prompt = load_prompt_from_file(system_prompt_path)
        self.user_prompt_template = load_prompt_from_file(user_prompt_template_path)

        if not self.system_prompt or not self.user_prompt_template:
            print("ServiceAnalysisAgent 경고: 프롬프트 파일 로드 실패. 기본 프롬프트를 사용하거나 기능이 제한될 수 있습니다.")
            # 대체 프롬프트 (실제 운영 시에는 더 견고한 오류 처리나 기본 프롬프트 설정 필요)
            self.system_prompt = "당신은 AI 서비스 분석가입니다. 제공된 정보를 바탕으로 서비스 특징을 JSON으로 요약해주세요. 요청된 모든 필드를 포함해야 합니다: service_name, description, core_features, target_users, collected_data_types, service_url_status, key_information_source."
            self.user_prompt_template = "서비스 URL: {service_url}\n문서 경로: {document_paths}\n항목별 RAG 컨텍스트:\n{rag_context}\n\n위 정보를 종합하여 JSON으로 분석 결과를 알려주세요."

        # RAG 쿼리를 생성할 정보 항목 정의
        self.info_items_for_rag = {
            "service_name": "서비스의 공식 명칭 또는 이름",
            "description": "서비스에 대한 전반적인 설명, 목적, 주요 가치",
            "core_features": "서비스가 제공하는 핵심 기능들",
            "target_users": "서비스를 주로 사용하는 대상 사용자 그룹 또는 고객층",
            "collected_data_types": "서비스가 사용자로부터 수집하거나 처리하는 데이터의 종류",
            "service_url_status": "제공된 서비스 URL의 현재 접속 상태 (예: 접속 가능, 리디렉션, 오류 등)",
            "key_information_source": "서비스 정보를 얻을 수 있는 주요 출처 (문서명, 웹페이지 섹션 등)"
        }

    def _get_single_item_rag_context(self, item_description: str, service_url: str, documents_to_consider: List[str]) -> str:
        """단일 정보 항목에 대한 RAG 컨텍스트를 가져옵니다."""
        if not self.retriever:
            return f"  - {item_description}: Retriever가 제공되지 않아 컨텍스트를 가져올 수 없습니다.\n"

        # RAG 쿼리 구성
        query = f"'{service_url}' 서비스의 '{item_description}'에 대한 정보를 관련 문서에서 찾아주세요."
        if documents_to_consider:
            doc_names = ", ".join([os.path.basename(doc_path) for doc_path in documents_to_consider])
            query += f" (주요 참고 문서: {doc_names})"
        
        print(f"ServiceAnalysisAgent: RAG 쿼리 (항목: {item_description}) - \"{query}\"")
        
        try:
            if hasattr(self.retriever, 'invoke'):
                relevant_docs = self.retriever.invoke(query)
            elif hasattr(self.retriever, 'get_relevant_documents'):
                relevant_docs = self.retriever.get_relevant_documents(query)
            else:
                return f"  - {item_description}: Retriever 메소드 오류로 컨텍스트를 가져올 수 없습니다.\n"
        except Exception as e:
            return f"  - {item_description}: RAG 컨텍스트 검색 중 오류 발생 ({e})\n"
            
        context_str = f"  - '{item_description}'에 대한 RAG 검색 결과:\n"
        if not relevant_docs:
            context_str += "    관련 문서를 찾을 수 없습니다.\n"
            return context_str

        for i, doc in enumerate(relevant_docs):
            source_file = doc.metadata.get('source_file', doc.metadata.get('source', 'N/A'))
            page_num = doc.metadata.get('page', 'N/A')
            section_title = doc.metadata.get('section_title', 'N/A')
            context_str += (
                f"    --- 컨텍스트 {i+1} (출처: {source_file}, 페이지: {page_num}, 섹션: {section_title}) ---\n"
                f"    {doc.page_content[:300]}...\n" # 컨텍스트 길이 제한
            )
        return context_str + "\n"

    def _get_comprehensive_rag_context(self, service_url: str, documents_to_consider: List[str]) -> str:
        """정의된 모든 정보 항목에 대해 RAG를 수행하고 통합된 컨텍스트를 생성합니다."""
        comprehensive_context = "## 항목별 RAG 컨텍스트 요약:\n"
        if not self.retriever:
            comprehensive_context += "Retriever가 제공되지 않아 RAG를 수행할 수 없습니다.\n"
            return comprehensive_context

        for item_key, item_desc_for_query in self.info_items_for_rag.items():
            # 실제 LLM에게 전달될 프롬프트에는 item_key가 아니라 item_desc_for_query가 더 유용할 수 있음
            # 여기서는 item_desc_for_query를 RAG 쿼리 생성에 사용
            context_for_item = self._get_single_item_rag_context(item_desc_for_query, service_url, documents_to_consider)
            comprehensive_context += context_for_item
        
        return comprehensive_context

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 실행"""
        print("ServiceAnalysisAgent 실행 시작 (항목별 RAG 적용)...")
        service_url = state.get("service_url", "")
        documents_paths = state.get("documents", []) 
        if not isinstance(documents_paths, list):
            print(f"ServiceAnalysisAgent 경고: 'documents'는 리스트여야 합니다. 현재 타입: {type(documents_paths)}")
            documents_paths = []

        # 각 정보 항목에 대해 RAG를 수행하여 통합 컨텍스트 생성
        rag_context = self._get_comprehensive_rag_context(service_url, documents_paths)
        
        document_paths_str = "\n".join([f"- {os.path.basename(path)}" for path in documents_paths]) if documents_paths else "제공된 문서 없음"

        human_prompt = self.user_prompt_template.format(
            service_url=service_url,
            document_paths=document_paths_str,
            rag_context=rag_context # 통합된, 항목별 RAG 컨텍스트
        )
        
        print("ServiceAnalysisAgent: LLM 호출 중...")
        # print(f"System Prompt:\n{self.system_prompt}") # 디버깅용
        # print(f"Human Prompt:\n{human_prompt}") # 디버깅용
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = self.llm.invoke(messages)
        
        service_info = {}
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                service_info = json.loads(json_str)
                print("ServiceAnalysisAgent: LLM으로부터 JSON 응답 파싱 성공.")
            else:
                print("ServiceAnalysisAgent 경고: LLM 응답에서 명확한 JSON 블록을 찾지 못했습니다. 전체 응답 파싱 시도.")
                # 전체 응답을 파싱하려고 시도하기 전에, 응답이 실제로 JSON인지 확인하는 것이 좋음
                # 여기서는 일단 시도하지만, 실제로는 더 견고한 오류 처리가 필요
                try:
                    service_info = json.loads(response.content)
                    print("ServiceAnalysisAgent: LLM 전체 응답 파싱 성공 (주의 필요).")
                except json.JSONDecodeError:
                    print("ServiceAnalysisAgent 오류: LLM 전체 응답도 JSON 형식이 아닙니다.")
                    raise # 원래 오류를 다시 발생시켜 호출 스택으로 전파
        except json.JSONDecodeError as e:
            print(f"ServiceAnalysisAgent 오류: LLM 응답 JSON 파싱 실패 - {e}")
            print(f"LLM 원본 응답 (일부):\n{response.content[:1000]}...") # 너무 긴 응답은 잘라서 출력
            service_info = {
                "error_message": "LLM 응답을 JSON으로 파싱하는 데 실패했습니다.",
                "service_name": "정보 추출 실패",
                "description": "정보 추출 실패",
                "core_features": [], "target_users": [], "collected_data_types": [],
                "service_url_status": "확인 불가", "key_information_source": "정보 추출 실패"
            }
        except Exception as e:
            print(f"ServiceAnalysisAgent 오류: 예기치 않은 오류 발생 - {e}")
            print(f"LLM 원본 응답 (일부):\n{response.content[:1000]}...")
            service_info = {"error_message": str(e)}

        state["service_info"] = service_info
        print(f"ServiceAnalysisAgent: 분석된 서비스 정보 (오류 포함 가능) - {service_info}")
        print("ServiceAnalysisAgent 실행 완료.")
        return state

