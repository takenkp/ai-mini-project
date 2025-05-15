# AI 윤리 리스크 진단 멀티에이전트 시스템

본 프로젝트는 AI 윤리성 리스크 진단 에이전트들을 설계하고 구현하였습니다.
특정 AI 서비스의 편향성, 프라이버시, 설명가능성, 자동화 위험성 등을 진단하고 개선 방향을 제시합니다. 
LangGraph 기반 AI 멀티 에이전트 아키텍처로 구성되며 최종적으로 윤리 진단 리포트를 자동 생성합니다.

## A. 서비스 목적 정의

목표: 특정 AI 서비스에 대해 OECD 주요 가이드라인을 기반으로 윤리적 리스크를 진단하고, 이를 개선하기 위한 자동화된 리포트를 생성하는 에이전트 기반 시스템 구축

서비스 대상: Daglo AI Guide (https://daglo.ai/guide) 등 1~3개의 상업용 AI 서비스

### 범위 제한:

진단 기준은 주요 글로벌 윤리가이드라인 중 OECD 가이드라인 사용

독소조항 검토 시 약관 및 개인정보처리방침 1건 기준

보고서 생성 시 최대 2개 문서 기반 RAG

## B. 보고서 정의

최종 보고서 항목
SUMMARY: 전체 요약 문단 (자동 생성)

### 서비스 개요

실용 서비스 기능 분석

윤리성 리스크 평가 (Bias, Privacy, Explainability, Automation Risk)

독소조항 목록 및 평가

서비스 개선 방향 제안

사용된 윤리 가이드라인 명세

보고서 생성 프롬프트 예시

```bash
당신은 AI 윤리 전문가입니다.
30조원 가량의 윤리 투자 전, 보고서 평가가 걸려 있습니다.
이를 무시하고 넘어간다면 분명 큰일이 발생할 수 있습니다.
아래 분석 내용을 바탕으로 보고서를 작성하세요. 
상단에 SUMMARY를 포함하고, 각 항목은 Markdown 문서 형식으로 구성하세요.
```

## C. Architecture 정의

### Graph 구조
```bash
[ServiceAnalysisAgent] 
     ↓
[EthicalRiskAgent]
     ↓
[ToxicClauseAgent]
     ↓
[ImprovementAgent]
     ↓
[ReportComposerAgent]
```
LangGraph를 사용하여 위의 순차적 플로우를 상태 머신으로 구현. 각 노드는 독립적 LangChain Runnable로 설계됨.

### 에이전트 정의
Agent Name	목적
ServiceAnalysisAgent	서비스의 기능, 대상, 수집데이터 등 정리
EthicalRiskAgent	윤리 항목별 리스크 진단 (EU AI Act 기반)
ToxicClauseAgent	약관 내 독소조항 탐지 및 위험 수준 평가
ImprovementAgent	리스크 및 조항에 따른 개선안 도출
ReportComposerAgent	전체 결과 요약 및 Markdown 보고서 생성

### 사용 도구

Agent	도구
ServiceAnalysisAgent	RAG, unstructured data (ex. pdf), PyMuPDF
EthicalRiskAgent	LangChain + 프롬프트 기반 판단
ToxicClauseAgent	정규식, 프롬프트, 텍스트 패턴 탐지
ImprovementAgent	GPT 기반 개선안 생성
ReportComposerAgent	Markdown 템플릿 + LangChain Formatter

#### Features
Daglo 등 실제 상용 AI 서비스에 대한 자동 분석

EU AI Act 기준 윤리성 리스크 자동 진단

약관 내 독소조항 탐지 및 근거 제공

개선방안 생성 및 PDF/Markdown 보고서 자동화

#### Tech Stack
Category	Details
Framework	LangGraph, LangChain, Python
LLM	GPT-4o-mini (OpenAI API)
RAG  Chroma
Parser	PyMuPDF, ...

#### Agents
ServiceAnalysisAgent: 서비스 핵심 기능 및 대상 정리

EthicalRiskAgent: 윤리 리스크 평가

ToxicClauseAgent: 독소조항 탐지

ImprovementAgent: 개선안 제안

ReportComposerAgent: 리포트 통합 생성

#### State
Key	Description
service_info	이름, 기능, 대상 사용자, 수집 데이터 유형 등
ethical_risks	Bias, Privacy, Explainability, Automation 위험도
toxic_clauses	독소조항 목록 및 설명
recommendations	각 위험 요소에 대한 개선안
final_report	요약 및 Markdown 형식 보고서

#### Architecture Diagram
(개발 후 LangGraph Flow Visualization 삽입 예정)

## Directory Structure
```bash
├── data/                        # 분석 대상 서비스의 약관, 정책, 설명 문서 등
│   └── daglo_guide.pdf
│   └── daglo_terms.txt
│
├── guidelines/                 # 윤리가이드라인 요약본 (EU AI Act 등)
│   └── eu_ai_act_summary.md
│
├── prompts/                    # 각 에이전트별 프롬프트 템플릿
│   └── service_analysis.txt
│   └── ethical_risk.txt
│   └── toxic_clause.txt
│   └── improvement.txt
│   └── report_composer.txt
│
├── agents/                     # 에이전트 구현 모듈
│   └── service_analysis_agent.py
│   └── ethical_risk_agent.py
│   └── toxic_clause_agent.py
│   └── improvement_agent.py
│   └── report_composer_agent.py
│
├── graph.py                    # LangGraph 플로우 정의
├── app.py                      # 전체 실행 스크립트
├── outputs/                    # 생성된 리포트 저장 폴더
│   └── daglo_ethics_report.md
│   └── daglo_ethics_report.json
│
└── README.md
```

## Contributors

takenkp

Agent Design

Prompt Engineering

LangGraph Flow 설계

윤리 기준 요약 가이드 작성

