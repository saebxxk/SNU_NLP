"""
router.py - Task Router
사전 정의된 Action Set(Task Catalog) 기반 질의 분류 및 실행
핵심 독창성: TF-IDF 유사도 + 슬롯 추출로 LLM 의존 없이 정확한 라우팅
"""
import re
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from db_schema import get_connection, DB_PATH
from tools import (
    resolve_metric, sql_fetch_fact, sql_fetch_series, sql_fetch_audit_opinion,
    sql_search_chunks, sql_find_extreme, sql_fetch_all_facts_for_year,
    calculate_change_rate, calculate_ratio, list_available_data,
    build_year_summary, format_value, RATIO_TYPES,
    sql_fetch_note_title, calculate_absolute_diff, get_metric_source,
)


# ─── 유사도 기반 라우터 ────────────────────────────────────────
class TFIDFRouter:
    """
    Task Catalog의 example_queries와 TF-IDF 코사인 유사도로
    사용자 질의를 가장 적합한 Task에 매핑
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.tasks = []
        self.corpus = []     # (task_name, query_text)
        self.tfidf_vectors = []
        self._load_tasks()

    def _load_tasks(self):
        """DB에서 Task Catalog 로드"""
        conn = get_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT task_name, task_group, description, "
                    "required_slots, execution_plan, response_template, example_queries "
                    "FROM task_catalog")
        rows = cur.fetchall()
        conn.close()

        self.tasks = []
        self.corpus = []

        for row in rows:
            task = {
                "task_name": row["task_name"],
                "task_group": row["task_group"],
                "description": row["description"],
                "required_slots": json.loads(row["required_slots"] or "[]"),
                "execution_plan": json.loads(row["execution_plan"] or "[]"),
                "response_template": row["response_template"] or "",
            }
            self.tasks.append(task)

            # 예시 질의 수집
            examples = json.loads(row["example_queries"] or "[]")
            # description도 포함
            all_text = " ".join(examples) + " " + row["description"]
            self.corpus.append((row["task_name"], all_text))

        # TF-IDF 벡터 계산
        self._build_tfidf()

    def _tokenize(self, text: str) -> list:
        """한국어 토크나이저 (간단한 n-gram + 키워드)"""
        tokens = []
        # 공백 분리
        words = re.split(r'\s+', text.strip())
        tokens.extend(words)
        # 2-gram
        for i in range(len(words) - 1):
            tokens.append(words[i] + words[i+1])
        return [t for t in tokens if t]

    def _build_tfidf(self):
        """TF-IDF 벡터 구축"""
        # 전체 문서 토크나이즈
        docs = []
        for _, text in self.corpus:
            docs.append(self._tokenize(text))

        # IDF 계산
        n_docs = len(docs)
        df = Counter()
        for doc_tokens in docs:
            for token in set(doc_tokens):
                df[token] += 1

        self.idf = {token: math.log((n_docs + 1) / (cnt + 1)) + 1
                    for token, cnt in df.items()}

        # TF-IDF 벡터
        self.tfidf_vectors = []
        for doc_tokens in docs:
            tf = Counter(doc_tokens)
            vec = {t: (cnt / len(doc_tokens)) * self.idf.get(t, 1)
                   for t, cnt in tf.items()}
            self.tfidf_vectors.append(vec)

    def _cosine_sim(self, vec_a: dict, vec_b: dict) -> float:
        """코사인 유사도"""
        keys = set(vec_a) & set(vec_b)
        if not keys:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in keys)
        norm_a = math.sqrt(sum(v**2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v**2 for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def route(self, query: str, top_k: int = 3) -> dict:
        """질의 → 가장 적합한 Task 반환"""
        q_tokens = self._tokenize(query)
        q_tf = Counter(q_tokens)
        q_vec = {t: (cnt / len(q_tokens)) * self.idf.get(t, 1)
                 for t, cnt in q_tf.items() if t}

        scores = []
        for i, vec in enumerate(self.tfidf_vectors):
            sim = self._cosine_sim(q_vec, vec)
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        best_idx, best_score = top[0]
        best_task = self.tasks[best_idx]

        return {
            "task": best_task,
            "confidence": round(best_score, 4),
            "top_matches": [
                {"task_name": self.tasks[i]["task_name"], "score": round(s, 4)}
                for i, s in top
            ]
        }


# ─── 슬롯 추출기 ──────────────────────────────────────────────
class SlotExtractor:
    """질의에서 year, metric, direction 등 슬롯 추출"""

    YEAR_RANGE = range(2014, 2025)

    # 연도 표현 패턴
    YEAR_PATTERNS = [
        r'(20\d{2})년',
        r'(20\d{2})\s*년도',
        r'(20\d{2})',
    ]

    # 비교 패턴
    COMPARE_PATTERNS = [
        r'(20\d{2})년.*?대비.*?(20\d{2})년',
        r'(20\d{2})년.*?vs.*?(20\d{2})년',
        r'(20\d{2})년.*?와.*?(20\d{2})년',
        r'(20\d{2})년.*?과.*?(20\d{2})년',
    ]

    DIRECTION_KEYWORDS = {
        "max": ["가장 높", "최고", "최대", "제일 많", "최고치", "최상"],
        "min": ["가장 낮", "최소", "최저", "제일 적", "최저치", "최하"],
    }

    RATIO_KEYWORDS = {
        "debt_ratio": ["부채비율"],
        "roe": ["roe", "자기자본이익률", "자본수익률"],
        "roa": ["roa", "총자산이익률", "자산수익률"],
        "equity_ratio": ["자기자본비율"],
        "operating_margin": ["영업이익률", "영업마진"],
    }

    # router.py
    # 질문에서 "주석 2", "주석3" 같은 note 번호를 추출한다.

    def _extract_note_number(self, query: str) -> int | None:
        """질문에서 주석 번호를 추출한다."""
        m = re.search(r"주석\s*(\d+)", query or "")
        if m:
            return int(m.group(1))
        return None

    def extract(self, query: str) -> dict:
        slots = {}

        # 연도 추출
        years = self._extract_years(query)
        # router.py
        # note_number 추출
        note_number = self._extract_note_number(query)
        if note_number is not None:
            slots["note_number"] = note_number
        if len(years) >= 2:
            slots["year_a"] = years[0]
            slots["year_b"] = years[-1]
        elif len(years) == 1:
            slots["year"] = years[0]

        # metric 추출
        metric_ko, metric_norm = resolve_metric(query)
        if metric_norm:
            slots["metric_ko"] = metric_ko
            slots["metric_norm"] = metric_norm

        # direction 추출
        for direction, keywords in self.DIRECTION_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                slots["direction"] = direction
                break

        # 재무비율 타입
        for ratio_type, keywords in self.RATIO_KEYWORDS.items():
            if any(kw in query.lower() for kw in keywords):
                slots["ratio_type"] = ratio_type
                break

        # 연도 범위 (추이/트렌드)
        trend_keywords = ["추이", "트렌드", "변화", "연도별", "연간", "시계열"]
        if any(kw in query for kw in trend_keywords):
            slots["is_trend"] = True
            if "year_a" not in slots and "year" not in slots:
                slots["start_year"] = 2014
                slots["end_year"] = 2024
            elif "year_a" in slots:
                slots["start_year"] = slots["year_a"]
                slots["end_year"] = slots["year_b"]

        # 키워드 검색어
        search_keywords = ["찾", "검색", "관련", "내용", "언급"]
        if any(kw in query for kw in search_keywords):
            # 짧은 명사 추출 (간단한 방법)
            clean = re.sub(r'\s+', ' ', query).strip()
            slots["search_keyword"] = clean

        # 연도 범위 체크
        if "year" in slots and slots["year"] not in self.YEAR_RANGE:
            slots["year_out_of_range"] = True

        return slots

    def _extract_years(self, query: str) -> list:
        years = []
        for pattern in self.YEAR_PATTERNS:
            matches = re.findall(pattern, query)
            for m in matches:
                y = int(m)
                if 2000 <= y <= 2030:
                    years.append(y)

        # 중복 제거하되 순서 유지
        seen = set()
        unique = []
        for y in years:
            if y not in seen:
                seen.add(y)
                unique.append(y)
        return unique


# ─── Task Executor ─────────────────────────────────────────────
class TaskExecutor:
    """Task + Slots → 실행 결과 반환"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def execute(self, task: dict, slots: dict, query: str) -> dict:
        task_name = task["task_name"]
        group = task["task_group"]

        if group == "infeasible":
            return {
                "status": "infeasible",
                "message": task["response_template"],
                "suggestion": "2014~2024년 삼성전자 재무 데이터 기반 질문을 해 주세요.",
            }

        if group == "exception":
            return self._handle_exception(task, slots, query)

        # feasible 실행
        try:
            return self._execute_feasible(task_name, slots, query)
        except Exception as e:
            return {"status": "error", "message": f"실행 오류: {str(e)}"}

    def _handle_exception(self, task: dict, slots: dict, query: str) -> dict:
        if task["task_name"] == "ambiguous_year_missing":
            return {
                "status": "exception",
                "message": "연도를 지정해 주세요. 예: '2023년 매출액은?'\n조회 가능 연도: 2014~2024년",
            }
        if task["task_name"] == "out_of_range_year":
            year = slots.get("year", "")
            return {
                "status": "exception",
                "message": f"요청하신 연도({year})는 보유 데이터 범위(2014~2024)를 벗어납니다.",
            }
        return {"status": "exception", "message": task["response_template"]}

    def _execute_feasible(self, task_name: str, slots: dict, query: str) -> dict:
        if task_name == "get_metric_by_year":
            year = slots.get("year")
            metric_norm = slots.get("metric_norm")
            if not year:
                return {"status": "exception", "message": "연도를 지정해 주세요."}
            if not metric_norm:
                return {"status": "exception", "message": "조회할 재무 지표를 지정해 주세요."}

            result = sql_fetch_fact(year, metric_norm, db_path=self.db_path)
            if result["value"] is None:
                return {
                    "status": "no_data",
                    "message": f"{year}년 {slots.get('metric_ko', metric_norm)} 데이터를 찾을 수 없습니다.",
                }
            return {
                "status": "success",
                "data": result,
                "answer": f"{year}년 {result['metric_ko']}은(는) {result['value_fmt']}입니다.",
            }

        elif task_name == "compare_metric_between_years":
            year_a = slots.get("year_a")
            year_b = slots.get("year_b")
            metric_norm = slots.get("metric_norm")
            if not year_a or not year_b:
                return {"status": "exception", "message": "비교할 두 연도를 지정해 주세요."}
            if not metric_norm:
                return {"status": "exception", "message": "비교할 재무 지표를 지정해 주세요."}

            res_a = sql_fetch_fact(year_a, metric_norm, db_path=self.db_path)
            res_b = sql_fetch_fact(year_b, metric_norm, db_path=self.db_path)
            change = calculate_change_rate(res_a.get("value"), res_b.get("value"))

            metric_ko = res_a.get("metric_ko") or slots.get("metric_ko", metric_norm)
            answer = (
                f"{year_a}년 대비 {year_b}년 {metric_ko}:\n"
                f"  • {year_a}년: {res_a['value_fmt']}\n"
                f"  • {year_b}년: {res_b['value_fmt']}\n"
            )
            if change.get("change_rate") is not None:
                answer += f"  • 증감률: {change['change_rate']:+.1f}% ({change.get('change_label', '')})"

            return {"status": "success", "data": {"a": res_a, "b": res_b, "change": change},
                    "answer": answer}

        elif task_name == "trend_of_metric":
            metric_norm = slots.get("metric_norm")
            if not metric_norm:
                return {"status": "exception", "message": "추이를 볼 재무 지표를 지정해 주세요."}

            start = slots.get("start_year", 2014)
            end = slots.get("end_year", 2024)
            series = sql_fetch_series(metric_norm, start, end, self.db_path)

            if not series:
                return {"status": "no_data", "message": f"{metric_norm} 시계열 데이터 없음"}

            metric_ko = series[0]["metric_ko"]
            lines = [f"{metric_ko} 연도별 추이:"]
            for item in series:
                lines.append(f"  {item['year']}년: {item['value_fmt']}")

            # 최고/최저
            valid = [s for s in series if s["value"] is not None]
            if valid:
                max_item = max(valid, key=lambda x: x["value"])
                min_item = min(valid, key=lambda x: x["value"])
                lines.append(f"\n  📈 최고: {max_item['year']}년 ({max_item['value_fmt']})")
                lines.append(f"  📉 최저: {min_item['year']}년 ({min_item['value_fmt']})")

            return {"status": "success", "data": series,
                    "answer": "\n".join(lines)}

        elif task_name == "find_best_worst_year":
            metric_norm = slots.get("metric_norm")
            direction = slots.get("direction", "max")
            if not metric_norm:
                return {"status": "exception", "message": "지표를 지정해 주세요."}

            result = sql_find_extreme(metric_norm, direction, self.db_path)
            if not result:
                return {"status": "no_data", "message": "데이터 없음"}

            answer = (f"{result['metric_ko']}이(가) {result['direction']} 해는 "
                      f"{result['year']}년으로 {result['value_fmt']}였습니다.")
            return {"status": "success", "data": result, "answer": answer}

        elif task_name == "get_metric_source":
            metric_norm = slots.get("metric_norm")
            ko_name = slots.get("metric_ko") or metric_norm
            if not metric_norm:
                return {"status": "exception", "message": "지표를 지정해 주세요."}
            return get_metric_source(metric_norm, ko_name)

        elif task_name == "calculate_metric_diff":
            year_a = slots.get("year_a")
            year_b = slots.get("year_b")
            metric_norm = slots.get("metric_norm")
            ko_name = slots.get("metric_ko") or metric_norm

            if not all([year_a, year_b, metric_norm]):
                return {"status": "exception", "message": "비교 연도 두 개와 지표를 모두 지정해 주세요."}

            result = calculate_absolute_diff(year_a, year_b, metric_norm, ko_name, self.db_path)
            return result

        elif task_name == "get_note_title_by_year_and_number":
            year = slots.get("year")
            note_number = slots.get("note_number")

            if year is None:
                return {"status": "exception", "message": "연도를 지정해 주세요."}
            if note_number is None:
                return {"status": "exception", "message": "주석 번호를 지정해 주세요."}

            result = sql_fetch_note_title(year, note_number, self.db_path)

            if not result.get("note_title"):
                return {
                    "status": "no_data",
                    "message": f"{year}년 주석 {note_number}의 제목을 찾을 수 없습니다.",
                }

            return {
                "status": "success",
                "data": result,
                "answer": result["note_title"],
            }

        elif task_name == "get_audit_opinion":
            year = slots.get("year")
            if not year:
                # 최근 연도 기본값
                year = 2024
            result = sql_fetch_audit_opinion(year, self.db_path)
            answer = f"{year}년 감사의견: {result.get('opinion', '데이터 없음')}"
            return {"status": "success", "data": result, "answer": answer}

        elif task_name == "search_by_keyword":
            keyword = slots.get("search_keyword") or query
            year = slots.get("year")
            chunks = sql_search_chunks(keyword, year, self.db_path)

            if not chunks:
                return {"status": "no_data",
                        "message": f"'{keyword}' 관련 내용을 찾을 수 없습니다."}

            lines = [f"'{keyword}' 검색 결과:"]
            for c in chunks:
                lines.append(f"\n[{c['year']}년 / {c['section_type']}]")
                lines.append(c["text"][:200] + "...")

            return {"status": "success", "data": chunks,
                    "answer": "\n".join(lines)}

        elif task_name == "calculate_financial_ratio":
            year = slots.get("year")
            ratio_type = slots.get("ratio_type")
            if not year:
                return {"status": "exception", "message": "연도를 지정해 주세요."}
            if not ratio_type:
                ratio_list = ", ".join(RATIO_TYPES.keys())
                return {"status": "exception",
                        "message": f"비율 유형을 지정해 주세요. 가능: {ratio_list}"}

            result = calculate_ratio(year, ratio_type, self.db_path)
            if "error" in result:
                return {"status": "error", "message": result["error"]}

            answer = f"{year}년 {result['ratio_name']}: {result['value_fmt']}"
            return {"status": "success", "data": result, "answer": answer}

        elif task_name == "list_available_data":
            data = list_available_data(self.db_path)
            answer = (
                f"조회 가능한 데이터:\n"
                f"• 연도: {', '.join(map(str, data['years']))}\n"
                f"• 재무 지표: {', '.join(data['metrics'][:10])} 등"
            )
            return {"status": "success", "data": data, "answer": answer}

        elif task_name == "summarize_year":
            year = slots.get("year")
            if not year:
                year = 2024
            summary = build_year_summary(year, self.db_path)
            return {"status": "success", "answer": summary}

        return {"status": "unknown", "message": f"알 수 없는 태스크: {task_name}"}


# ─── 메인 파이프라인 ──────────────────────────────────────────
class QueryPipeline:
    """
    사용자 질의 → Task 분류 → 슬롯 추출 → 실행 → 결과 반환
    LLM Action Set 기반 정교한 처리
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.router = TFIDFRouter(db_path)
        self.extractor = SlotExtractor()
        self.executor = TaskExecutor(db_path)

    # ─── 규칙 기반 우선 라우팅 ────────────────────────────────
    KEYWORD_RULES = [
        # (조건함수, task_name)  ← 우선순위 높은 것이 앞

        # 감사의견
        (lambda q, s: "감사의견" in q,
         "get_audit_opinion"),

        # 출처 재무제표 조회 (evidence) ← 가장 우선순위 높게
        (lambda q, s: (
            any(k in q for k in ["직접 근거", "근거가 되는 재무제표", "근거 재무제표", "어느 재무제표", "재무제표는 어디"])
            and s.get("metric_norm")
        ), "get_metric_source"),

        # 주석 N 주제 조회 (반드시 일찍 처리 - 숫자+주석 조합)
        (lambda q, s: (
            s.get("year") is not None
            and s.get("note_number") is not None
            and any(k in q for k in ["주석", "주제", "다루", "내용"])
        ), "get_note_title_by_year_and_number"),

        # 재무비율
        (lambda q, s: any(r in q for r in [
            "부채비율", "ROE", "ROA", "roe", "roa",
            "자기자본비율", "영업이익률", "자본이익률"
        ]),
         "calculate_financial_ratio"),

        # 증감액 (절대값 차이) ← compare보다 앞에 위치
        (lambda q, s: (
            "증감액" in q
            and s.get("year_a") and s.get("year_b") and s.get("metric_norm")
        ),
         "calculate_metric_diff"),

        # 증감 방향 (증가/감소 여부)
        (lambda q, s: (
            any(k in q for k in ["증가했는가", "감소했는가", "증가 감소", "늘었는가", "줄었는가"])
            and s.get("year_a") and s.get("year_b") and s.get("metric_norm")
        ),
         "compare_metric_between_years"),

        # 연도 비교 (증감률)
        (lambda q, s: (
            ("비교" in q or "대비" in q or " vs " in q)
            and s.get("year_a") and s.get("year_b") and s.get("metric_norm")
            and "증감액" not in q
        ),
         "compare_metric_between_years"),

        # 추이 / 트렌드
        # 주의: "영업활동현금흐름"처럼 metric 이름 자체에 '흐름'이 들어가므로
        # '흐름'을 trend 키워드로 쓰면 안 된다.
        (lambda q, s: (
            any(k in q for k in ["추이", "트렌드", "변화", "연도별", "시계열"])
            and s.get("metric_norm")
        ),
         "trend_of_metric"),

        # 최고/최저
        (lambda q, s: (
            any(k in q for k in [
                "가장 높", "최고", "최대", "제일 많",
                "가장 낮", "최소", "최저", "제일 적"
            ])
            and s.get("metric_norm")
        ),
         "find_best_worst_year"),

        # 연도 요약
        (lambda q, s: (
            any(k in q for k in ["요약", "정리", "분석", "현황"])
            and s.get("year")
        ),
         "summarize_year"),

        # 사용 가능 데이터 목록
        (lambda q, s: any(k in q for k in [
            "어떤 데이터", "무슨 질문", "어떤 연도",
            "어떤 지표", "가능", "도움", "도움말"
        ]),
         "list_available_data"),

        # 투자/주가 질문
        # 주의: "투자활동현금흐름"처럼 metric이 이미 잡힌 경우는
        # stock_recommendation으로 보내면 안 된다.
        (lambda q, s: (
            any(k in q for k in ["주가", "매수", "매도", "추천"])
            or ("투자" in q and not s.get("metric_norm"))
        ),
         "stock_recommendation"),

        # 미래 예측
        (lambda q, s: any(k in q for k in ["예측", "전망", "예상", "미래"]),
         "future_forecast"),

        # 경쟁사 비교
        (lambda q, s: (
            any(k in q for k in ["경쟁사", "비교", "하이닉스", "tsmc", "인텔", "업계"])
            and not s.get("year_a")
        ),
         "competitor_comparison"),

        # 연도 + metric → 단일 조회
        (lambda q, s: (
            s.get("year") and s.get("metric_norm")
            and not any(k in q for k in ["비교", "대비", "추이", "트렌드", "변화", "연도별", "시계열"])
        ),
         "get_metric_by_year"),

        # 연도만 있고 metric 없음 → 요약
        (lambda q, s: (
            s.get("year") and not s.get("metric_norm")
        ),
         "summarize_year"),
    ]

    def _keyword_route(self, query: str, slots: dict):
        """키워드 기반 우선 라우팅 → task_name 반환 (없으면 None)"""
        for condition, task_name in self.KEYWORD_RULES:
            try:
                if condition(query, slots):
                    # task_catalog에서 task 객체 찾기
                    for task in self.router.tasks:
                        if task["task_name"] == task_name:
                            return task, 1.0
                    # 카탈로그에 없는 task도 허용 (calculate_metric_diff 등)
                    return {"task_name": task_name, "task_group": "feasible",
                            "response_template": ""}, 1.0
            except Exception:
                pass
        return None, 0.0

    def process(self, query: str, verbose: bool = False) -> dict:
        """전체 처리 파이프라인"""

        # 1. 슬롯 추출 (LLM 전 규칙 기반)
        slots = self.extractor.extract(query)

        # 2. 연도 범위 체크 (예외 처리)
        if slots.get("year_out_of_range"):
            return {
                "query": query,
                "task_name": "out_of_range_year",
                "task_group": "exception",
                "slots": slots,
                "result": {"status": "exception",
                           "message": f"요청 연도({slots.get('year')})는 2014~2024 범위를 벗어납니다."},
            }

        # 3. 키워드 기반 우선 라우팅 (정확도 높음)
        keyword_task, keyword_conf = self._keyword_route(query, slots)

        # 4. TF-IDF 라우팅 (폴백)
        route = self.router.route(query)
        tfidf_task = route["task"]
        tfidf_conf = route["confidence"]

        # 규칙 기반이 있으면 우선 사용
        if keyword_task:
            task = keyword_task
            confidence = keyword_conf
            routing_method = "keyword"
        else:
            task = tfidf_task
            confidence = tfidf_conf
            routing_method = "tfidf"

        if verbose:
            print(f"  🔍 라우팅: {task['task_name']} (신뢰도: {confidence:.3f}, 방법: {routing_method})")
            print(f"  📋 슬롯: {slots}")

        # 5. 신뢰도가 너무 낮으면 ambiguous 처리
        if confidence < 0.01 and task["task_group"] == "feasible":
            task = {"task_name": "ambiguous_year_missing",
                    "task_group": "exception",
                    "response_template": "질문을 이해하지 못했습니다. 더 구체적으로 질문해 주세요."}

        # 6. 실행
        result = self.executor.execute(task, slots, query)

        return {
            "query": query,
            "task_name": task["task_name"],
            "task_group": task["task_group"],
            "slots": slots,
            "confidence": confidence,
            "result": result,
        }

    def get_action_set_summary(self) -> str:
        """LLM이 로드하는 Action Set 요약 텍스트"""
        lines = ["【사전 정의된 Action Set (Task Catalog)】\n"]
        for task in self.router.tasks:
            group_emoji = {"feasible": "✅", "infeasible": "❌", "exception": "⚠️"}.get(
                task["task_group"], "•")
            lines.append(f"{group_emoji} {task['task_name']}: {task['description']}")
        return "\n".join(lines)


if __name__ == "__main__":
    pipeline = QueryPipeline()
    print(pipeline.get_action_set_summary())

    test_queries = [
        "2023년 매출액은?",
        "2019년과 2023년 영업이익 비교",
        "최근 10년 자산총계 추이",
        "2023년 부채비율 계산해줘",
        "삼성전자 주식 사도 돼?",
        "2025년 실적 예측해줘",
    ]

    print("\n=== 라우팅 테스트 ===")
    for q in test_queries:
        result = pipeline.process(q, verbose=True)
        print(f"Q: {q}")
        print(f"→ {result['result'].get('answer', result['result'].get('message', ''))}\n")
