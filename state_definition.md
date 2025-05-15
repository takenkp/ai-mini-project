## Agent Input/Output Schema

---

### 1. **ServiceAnalysisAgent**

#### 📥 Input

```json
{
  "service_url": "https://daglo.ai/guide",
  "documents": ["./data/daglo_guide.pdf"]
}
```

#### 📤 Output

```json
{
  "service_name": "Daglo AI Guide",
  "description": "사용자가 입력한 문서를 요약해주는 대화형 AI 가이드",
  "core_features": ["문서 요약", "인터랙티브 가이드", "챗 기반 추천"],
  "target_users": ["기업 사용자", "일반 사용자"],
  "collected_data_types": ["텍스트 입력", "클릭 로그"]
}
```

---

### 2. **EthicalRiskAgent**

#### 📥 Input

```json
{
  "service_info": {
    "service_name": "Daglo AI Guide",
    "core_features": ["문서 요약", "인터랙티브 가이드"],
    "collected_data_types": ["텍스트 입력", "클릭 로그"]
  },
  "guideline": "EU AI Act"
}
```

#### 📤 Output

```json
{
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
```

---

### 3. **ToxicClauseAgent**

#### 📥 Input

```json
{
  "terms_text": "...(약관 원문 또는 URL)",
  "privacy_policy_text": "...(개인정보 처리방침 원문 또는 URL)"
}
```

#### 📤 Output

```json
{
  "toxic_clauses": [
    {
      "clause": "서비스 제공자는 사전 통보 없이 서비스를 중단할 수 있음",
      "risk_reason": "사용자 권리에 대한 과도한 제한"
    },
    {
      "clause": "사용자 데이터는 마케팅 목적으로 무제한 활용될 수 있음",
      "risk_reason": "프라이버시 침해 우려"
    }
  ],
  "overall_clause_risk": "높음"
}
```

---

### 4. **ImprovementAgent**

#### 📥 Input

```json
{
  "ethical_risks": {
    "bias_risk": "중간",
    "privacy_risk": "높음",
    "explainability_risk": "낮음",
    "automation_risk": "중간"
  },
  "toxic_clauses": [
    "서비스 제공자는 사전 통보 없이 서비스를 중단할 수 있음",
    "사용자 데이터는 마케팅 목적으로 무제한 활용될 수 있음"
  ]
}
```

#### 📤 Output

```json
{
  "recommendations": {
    "bias_risk": "사용자 그룹별 테스트 및 결과 로그 분석을 통해 편향 여부 검증",
    "privacy_risk": "데이터 최소 수집 및 삭제 주기 명시",
    "explainability_risk": "결과 생성 근거를 사용자에게 제공하는 UI 설계",
    "automation_risk": "자동화된 추천 결과에 수동 검토 옵션 추가",
    "toxic_clauses": "약관에서 사용자 권리를 침해하는 조항 제거 또는 명확화"
  }
}
```

---

### 5. **ReportComposerAgent**

#### 📥 Input

```json
{
  "service_info": {...},
  "ethical_risks": {...},
  "toxic_clauses": [...],
  "recommendations": {...}
}
```

#### 📤 Output

```json
{
  "summary": "Daglo AI Guide는 문서 요약 기반 AI 서비스로, 프라이버시 및 약관 관련 윤리 리스크가 다소 높게 평가됨. 설명가능성은 낮으나 편향성은 중간 수준. 주요 개선 방향은 데이터 최소 수집 및 약관 명확화.",
  "report_markdown": "./outputs/daglo_ethics_report.md",
  "report_json": "./outputs/daglo_ethics_report.json"
}
```
