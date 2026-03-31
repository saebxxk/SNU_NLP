# Samsung Audit Report Parser

삼성전자 감사보고서 HTML 파일을 파싱하여 다음 정보를 구조화하는 프로젝트입니다.

- 감사보고서 메타데이터 추출
- 재무제표 표 자동 분류
- 5대 재무제표 package 구성
- 재무제표 정규화
- 표 제거 일반 텍스트(`text_only`) 추출
- 검증 리포트 생성

---

## 1. 프로젝트 개요

이 프로젝트는 삼성전자 감사보고서 HTML 파일을 입력으로 받아,
감사보고서 내 표들을 자동 분류하고 재무제표를 정규화하는 것을 목표로 합니다.

현재 구현 범위는 다음과 같습니다.

- HTML 파일 로드
- 전체 텍스트 추출
- 표 제거 텍스트 추출
- HTML 표(`pd.read_html`) 추출
- 표 유형 분류
- 재무제표 5종 package 구성
- 재무제표 정규화
- 검증 리포트 출력

---

## 2. 프로젝트 구조

```text
report_parser/
├── __init__.py
├── audit_report_parser.py
├── models.py
├── run_parse.py
├── table_parsers
│   ├── __init__.py
│   ├── base.py
│   ├── misc.py
│   ├── note_tables.py
│   └── statement.py
├── text_parsers
│   ├── __init__.py
│   ├── extractor.py
│   ├── headings.py
│   └── sections.py
├── utils.py
└── validation.py
```

### 주요 파일 설명

- `audit_report_parser.py`  
  최상위 parser orchestration

- `models.py`  
  공통 데이터 클래스 정의

- `utils.py`  
  문자열/숫자 처리 공통 유틸 함수

- `validation.py`  
  파싱 결과 검증 및 디버깅 출력 함수

- `run_parse.py`  
  실행용 스크립트

### `table_parsers/`

- `base.py`  
  모든 표 parser의 공통 부모 클래스 및 feature 추출 로직

- `statement.py`  
  재무제표 제목표 / 본문표 / footer 표 parser

- `note_tables.py`  
  금융상품표, 롤포워드표, 배당표, 연금표 등 주석 내 특수 표 parser

- `misc.py`  
  note-like table, metadata table, unknown table 등 fallback parser

### `text_parsers/`

현재는 일반 텍스트 파싱 확장을 위한 구조만 마련되어 있습니다.

- `extractor.py`  
  텍스트 추출/분리 관련 로직

- `headings.py`  
  heading 탐지 로직

- `sections.py`  
  section grouping 로직

---

## 3. 설치 방법

### 3.1 Python 버전

권장 버전:

- Python 3.10+
- Python 3.11+
- Python 3.12+

### 3.2 의존성 설치

```bash
pip install -r requirements.txt
```

---

## 4. 실행 방법


python run_parse.py data/감사보고서_2024.htm
---

## 5. 출력 결과

parser 실행 결과는 dictionary 형태로 반환됩니다.

주요 키는 다음과 같습니다.

- `metadata`
- `document_structure`
- `sections`
- `tables`
- `raw`

### `metadata`

- source file
- encoding
- company name
- report date

### `document_structure`

- 전체 텍스트 길이
- 표 제거 텍스트 길이
- 표 개수
- 주요 section keyword 위치

### `tables`

- `summary`: 전체 표 요약 DataFrame
- `classified`: 표별 분류 결과
- `packages`: 재무제표 package 정보
- `normalized`: 정규화된 재무제표 결과
- `raw_tables`: 원본 표 리스트

### `raw`

- 원본 HTML
- 전체 텍스트
- 표 제거 텍스트

---

## 6. 현재 상태

현재 2024년 삼성전자 감사보고서 기준으로 다음이 확인되었습니다.

- 총 278개 표 추출
- `unknown_table = 0`
- 5대 재무제표 package 정상 구성
- normalized 결과 complete
- 자본변동표 block 분리 성공
- suspicious table 없음

---

## 7. 현재 한계

현재 프로젝트에는 다음 한계가 있습니다.

1. `note_like_table` 비중이 아직 큼  
   → 향후 더 세부적인 parser subclass 추가 필요

2. 일반 텍스트 파싱은 아직 확장 단계  
   → `text_parsers/`를 실제 parsing pipeline에 더 연결해야 함

3. 입력 파일 경로가 `run_parse.py`에 하드코딩되어 있음  
   → 향후 CLI argument 방식으로 개선 가능

4. 현재는 특정 샘플(2024) 기준 검증이 중심  
   → 2014–2024 전체 배치 검증이 필요

---

## 8. 다음 단계

향후 작업 우선순위는 다음과 같습니다.

1. `note_like_table` 세분화
2. 일반 텍스트 section parser 연결
3. 결과 JSON/CSV 저장 기능 추가
4. batch parsing 지원
5. 테스트 코드 추가
6. GitHub Actions 기반 CI 구성

---

## 9. 예시 워크플로우

1. HTML 파일 로드
2. BeautifulSoup으로 문서 파싱
3. 전체 텍스트 / 표 제거 텍스트 추출
4. `pd.read_html()`로 모든 표 추출
5. 각 표에 적절한 parser class 할당
6. 재무제표 package 구성
7. normalized output 생성
8. validation report 출력

---

## 10. 작성 메모

이 프로젝트는 금융 도메인 HTML 감사보고서를 대상으로 하는
구조화 파싱 파이프라인 구축을 목표로 합니다.

특히 다음 문제를 해결하는 데 초점을 둡니다.

- 연도별 HTML 구조 차이 대응
- 재무제표 표 구조 일반화
- 감사보고서 텍스트/표 분리
- 이후 RAG 기반 QA 시스템 확장 가능성 확보
