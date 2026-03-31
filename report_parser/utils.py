# utils.py

import re
from typing import Any
import pandas as pd


def clean_cell(x: Any) -> str:
    # 셀 문자열 정리 함수
    if pd.isna(x):
        return ""

    s = str(x)
    s = s.replace("\xa0", " ")
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_for_keyword_match(text: Any) -> str:
    # 키워드 매칭용으로 공백 제거 및 소문자화
    return re.sub(r"\s+", "", str(text)).lower()


def table_to_text(df: pd.DataFrame) -> str:
    # 표 전체를 하나의 문자열로 합치는 함수
    values = df.fillna("").astype(str).values.ravel().tolist()
    values = [clean_cell(v) for v in values]
    return " ".join(v for v in values if v)


def summarize_table(df: pd.DataFrame, n_rows: int = 3, n_cols: int = 6) -> str:
    # 표 preview 생성 함수
    temp = df.iloc[:n_rows, :n_cols].copy()
    temp = temp.map(clean_cell)

    lines = []
    for _, row in temp.iterrows():
        lines.append(" | ".join(row.tolist()))

    return "\n".join(lines)


def is_numeric_like_string(x: Any) -> bool:
    # 숫자처럼 보이는 문자열인지 판별
    s = clean_cell(x)

    if s == "" or s == "-":
        return False

    s = s.replace(",", "")

    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]

    try:
        float(s)
        return True
    except ValueError:
        return False


def count_numeric_like_cells(df: pd.DataFrame) -> int:
    # 숫자처럼 보이는 셀 개수 계산
    count = 0

    for v in df.fillna("").astype(str).values.ravel():
        if is_numeric_like_string(v):
            count += 1

    return count


def parse_number(x: Any) -> Any:
    # 숫자 문자열을 숫자로 변환
    if pd.isna(x):
        return pd.NA

    s = clean_cell(x)

    if s == "" or s == "-":
        return pd.NA

    s = s.replace(",", "")

    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]

    try:
        num = float(s)
    except ValueError:
        return pd.NA

    if num.is_integer():
        return int(num)

    return num


def clean_note_value(x: Any) -> Any:
    # 주석 값 문자열 정리
    if pd.isna(x):
        return pd.NA

    s = clean_cell(x)

    if s == "":
        return pd.NA

    # 단일 숫자 float 꼴이면 정수 문자열로 정리
    try:
        num = float(s)
        if num.is_integer():
            return str(int(num))
    except ValueError:
        pass

    # 쉼표 분리 후 각 조각 정리
    parts = [p.strip() for p in s.split(",")]
    cleaned_parts = []

    for p in parts:
        try:
            num = float(p)
            if num.is_integer():
                cleaned_parts.append(str(int(num)))
            else:
                cleaned_parts.append(str(num))
        except ValueError:
            cleaned_parts.append(p)

    return ", ".join(cleaned_parts)