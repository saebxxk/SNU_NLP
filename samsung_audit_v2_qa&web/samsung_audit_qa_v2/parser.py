"""
parser.py - 삼성전자 감사보고서 HTML 파서
10개년(2014-2024) 감사보고서 HTML → 구조화된 데이터로 변환
"""
import re
import glob
import os
from bs4 import BeautifulSoup
from datetime import datetime


# ─────────────────────────────────────────────
# 재무 항목 정규화 사전 (한국어 → 영문 canonical)
# ─────────────────────────────────────────────
METRIC_MAP = {
    "매출액": "revenue",
    "영업수익": "revenue",
    "수익": "revenue",
    "영업이익": "operating_income",
    "영업이익(손실)": "operating_income",
    "당기순이익": "net_income",
    "당기순이익(손실)": "net_income",
    "순이익": "net_income",
    "자산총계": "total_assets",
    "자산 총계": "total_assets",
    "부채총계": "total_liabilities",
    "부채 총계": "total_liabilities",
    "자본총계": "total_equity",
    "자본 총계": "total_equity",
    "유동자산": "current_assets",
    "비유동자산": "non_current_assets",
    "유동부채": "current_liabilities",
    "비유동부채": "non_current_liabilities",
    "현금및현금성자산": "cash_and_equivalents",
    "현금 및 현금성자산": "cash_and_equivalents",
    "감가상각비": "depreciation",
    "무형자산상각비": "amortization",
    "연구개발비": "r_and_d",
    "법인세비용": "income_tax",
    "이자비용": "interest_expense",
    "영업활동현금흐름": "operating_cash_flow",
    "투자활동현금흐름": "investing_cash_flow",
    "재무활동현금흐름": "financing_cash_flow",
}

# 단위 정규화 (백만원 → KRW 백만)
UNIT_MAP = {
    "백만원": 1_000_000,
    "천원": 1_000,
    "억원": 100_000_000,
    "원": 1,
    "million": 1_000_000,
}

# 섹션 타입 키워드 매핑
SECTION_KEYWORDS = {
    "audit_opinion": ["감사의견", "독립된 감사인"],
    "balance_sheet": ["재무상태표", "대차대조표"],
    "income_statement": ["손익계산서", "포괄손익계산서"],
    "cash_flow": ["현금흐름표"],
    "equity_statement": ["자본변동표"],
    "notes": ["주석", "재무제표 주석"],
    "key_audit_matters": ["핵심감사사항"],
}


def normalize_number(value_str: str, unit_multiplier: float = 1.0):
    """문자열 숫자 → float 변환 (괄호 음수, 쉼표, 단위 처리)"""
    if value_str is None:
        return None
    s = str(value_str).strip().replace(",", "").replace(" ", "").replace("\xa0", "")
    if s in {"", "-", "—", "N/A", "해당없음"}:
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = s.replace("(", "").replace(")", "")
    try:
        val = float(s)
        if negative:
            val = -val
        return val * unit_multiplier
    except ValueError:
        return None


def detect_year_from_filename(filepath: str) -> int:
    """파일명에서 연도 추출 (감사보고서_2024.htm → 2024)"""
    m = re.search(r'(\d{4})', os.path.basename(filepath))
    return int(m.group(1)) if m else None


def detect_unit(text: str) -> tuple:
    """텍스트에서 단위 감지 → (단위명, 배수) 반환"""
    for unit, multiplier in UNIT_MAP.items():
        if unit in text:
            return unit, multiplier
    return "원", 1


def identify_section_type(text: str) -> str:
    """텍스트 내용으로 섹션 타입 분류"""
    for stype, keywords in SECTION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return stype
    return "other"


def extract_audit_opinion(soup: BeautifulSoup) -> str:
    """감사의견 섹션 텍스트 추출"""
    text = soup.get_text()
    # 감사의견 섹션 찾기
    idx = text.find("감사의견")
    if idx < 0:
        return ""
    # 적정/한정/부적정/의견거절 판단
    snippet = text[idx:idx+500]
    return snippet.strip()


def detect_opinion_type(opinion_text: str) -> str:
    """감사의견 유형 판별 (여러 표현 방식 처리)"""
    # 부적정/거절을 먼저 체크 (더 구체적인 것 우선)
    if "의견거절" in opinion_text:
        return "의견거절"
    if "부적정" in opinion_text:
        return "부적정의견"
    if "한정의견" in opinion_text:
        return "한정의견"
    # 구버전 형식: "공정하게 표시하고 있습니다" → 적정
    if "공정하게 표시하고 있습니다" in opinion_text:
        return "적정의견"
    # 신버전 형식: 명시적 "적정의견"
    if "적정의견" in opinion_text:
        return "적정의견"
    # 일반 적정 패턴
    if "적정하게" in opinion_text or ("적정" in opinion_text and "한정" not in opinion_text):
        return "적정의견"
    return "알 수 없음"


def clean_cell_text(text: str) -> str:
    """셀 텍스트 정규화: 내부 공백 제거, 특수문자 제거"""
    # 공백 통합 + trim
    s = re.sub(r'\s+', '', text)
    # 로마 숫자 접두어 제거 (Ⅰ. Ⅱ. 등)
    s = re.sub(r'^[Ⅰ-Ⅹ\d]+\.\s*', '', s)
    # 특수문자 제거
    s = s.replace('\xa0', '').replace('　', '')
    return s.strip()


def match_metric(label: str) -> tuple:
    """label 텍스트에서 metric 매칭 (정확도 우선 순위)"""
    clean = clean_cell_text(label)

    # 1순위: 정확 매칭 (클린 텍스트)
    if clean in METRIC_MAP:
        return clean, METRIC_MAP[clean]

    # 2순위: 원본 텍스트에서 부분 매칭 (긴 것 우선)
    sorted_keys = sorted(METRIC_MAP.keys(), key=len, reverse=True)
    for ko_name in sorted_keys:
        # 이자수익, 수익증권 등 잘못된 매칭 방지
        if ko_name == "수익" and any(x in clean for x in ["이자수익", "수익증권", "기타수익"]):
            continue
        if ko_name in clean:
            return ko_name, METRIC_MAP[ko_name]

    # 3순위: 원본 라벨에서 검색
    for ko_name in sorted_keys:
        if ko_name == "수익" and any(x in label for x in ["이자수익", "수익증권"]):
            continue
        if ko_name in label:
            return ko_name, METRIC_MAP[ko_name]

    return None, None


def extract_financial_tables(soup: BeautifulSoup, year: int) -> list:
    """재무 테이블 추출 → financial_fact 리스트 반환"""
    facts = []
    tables = soup.find_all("table")

    for table in tables:
        table_text = table.get_text()
        # 재무 관련 테이블만 처리 (키워드 기반 필터)
        clean_table = clean_cell_text(table_text)
        has_financial = any(kw in clean_table for kw in
                           ["매출액", "영업이익", "자산총계", "당기순이익", "부채총계"])
        if not has_financial:
            continue

        # 단위 감지
        unit_name, unit_mult = detect_unit(table_text)

        # 연결/별도 구분
        scope = "consolidated" if "연결" in table_text else "separate"

        # 테이블 타입 감지
        if "재무상태표" in clean_table or "자산총계" in clean_table:
            table_type = "balance_sheet"
        elif "손익" in clean_table or "매출액" in clean_table:
            table_type = "income_statement"
        elif "현금흐름" in clean_table:
            table_type = "cash_flow"
        else:
            table_type = "other"

        rows = table.find_all("tr")
        for row in rows:
            cells_raw = row.find_all(["td", "th"])
            if len(cells_raw) < 2:
                continue

            cells = [td.get_text().replace("\xa0", " ").strip()
                     for td in cells_raw]

            label = cells[0]
            ko_name, norm_label = match_metric(label)

            if norm_label is None:
                continue

            # 값 추출 - 숫자가 있는 첫 번째 셀 탐색
            # 주석 번호(1-2자리)는 건너뜀; 실제 재무 값(보통 3자리 이상)만 추출
            current_val = None
            for cell in cells[1:5]:  # 최대 5번째 셀까지
                cell_clean = cell.strip().replace(',', '').replace(' ', '').replace('\xa0','')
                # 짧은 숫자(주석번호 1-3자리)는 건너뜀
                if re.match(r'^\d{1,3}$', cell_clean):
                    continue
                val = normalize_number(cell, unit_mult)
                if val is not None and abs(val) > 100:  # 의미있는 크기의 값만
                    current_val = val
                    break

            if current_val is not None:
                facts.append({
                    "year": year,
                    "metric_name_ko": ko_name,
                    "metric_name_norm": norm_label,
                    "value": current_val,
                    "unit": unit_name,
                    "consolidation_scope": scope,
                    "statement_type": table_type,
                    "source_row_label": label.strip(),
                })

    # 중복 제거 (같은 연도+metric+scope에서 첫 번째 값 사용)
    seen = set()
    deduped = []
    for fact in facts:
        key = (fact["year"], fact["metric_name_norm"], fact["consolidation_scope"])
        if key not in seen:
            seen.add(key)
            deduped.append(fact)

    return deduped


def extract_text_chunks(soup: BeautifulSoup, year: int, chunk_size: int = 500) -> list:
    """본문 텍스트를 청크 단위로 분할 → 검색용"""
    text = soup.get_text()
    # 연속 공백/개행 정제
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    chunks = []
    # 섹션 단위로 분할
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]

    for i, para in enumerate(paragraphs):
        section_type = identify_section_type(para)
        chunks.append({
            "year": year,
            "chunk_index": i,
            "text": para[:chunk_size],
            "char_count": len(para),
            "section_type": section_type,
        })

    return chunks


def parse_audit_report(filepath: str) -> dict:
    """감사보고서 HTML 파일 → 구조화된 파싱 결과 반환"""
    year = detect_year_from_filename(filepath)
    if year is None:
        raise ValueError(f"연도를 파일명에서 추출할 수 없음: {filepath}")

    with open(filepath, "rb") as f:
        raw = f.read()

    soup = BeautifulSoup(raw, "html.parser", from_encoding="cp949")
    text = soup.get_text()

    # 감사의견
    opinion_text = extract_audit_opinion(soup)
    opinion_type = detect_opinion_type(opinion_text)

    # 재무 팩트
    financial_facts = extract_financial_tables(soup, year)

    # 텍스트 청크
    chunks = extract_text_chunks(soup, year)

    return {
        "year": year,
        "source_file": os.path.basename(filepath),
        "source_path": filepath,
        "parsed_at": datetime.now().isoformat(),
        "audit_opinion": opinion_type,
        "audit_opinion_text": opinion_text,
        "financial_facts": financial_facts,
        "chunks": chunks,
        "total_text_length": len(text),
    }


def parse_all_reports(data_dir: str) -> list:
    """디렉토리 내 모든 감사보고서 파싱 (glob 기반, 한글 경로 지원)"""
    # 직접 join 시 한글 경로 문제 방지: glob recursive 사용
    pattern1 = os.path.join(data_dir, "*.htm")
    pattern2 = os.path.join(data_dir, "**", "*.htm")
    files = sorted(glob.glob(pattern1) + glob.glob(pattern2, recursive=True))
    # fallback: 전역 glob
    if not files:
        all_files = glob.glob('/sessions/**/*.htm', recursive=True)
        files = sorted(all_files)

    results = []
    for fpath in files:
        year = detect_year_from_filename(fpath)
        if year:
            print(f"  파싱 중: {year}년 ({os.path.basename(fpath)})")
            try:
                result = parse_audit_report(fpath)
                results.append(result)
            except Exception as e:
                print(f"  ⚠️  {year}년 파싱 오류: {e}")

    print(f"\n총 {len(results)}개 파일 파싱 완료")
    return results


if __name__ == "__main__":
    import sys
    import json

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    results = parse_all_reports(data_dir)

    for r in results:
        print(f"{r['year']}년: 팩트 {len(r['financial_facts'])}건, "
              f"청크 {len(r['chunks'])}건, 감사의견: {r['audit_opinion']}")
