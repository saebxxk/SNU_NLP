"""
chatbot.py - 삼성전자 감사보고서 QA 챗봇 (CLI)
파이프라인 우선 + LLM 선택 옵션:
  1) Pipeline-only  (로컬 모델 없어도 동작, 기본값)
  2) Commercial API  (상용 LLM API_KEY 환경변수 설정 시)
  3) Ollama 로컬  (Ollama 설치 + 모델 다운로드 시 자동 탐지)
"""
import os
import sys
import json
import uuid
import re
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from db_schema import get_connection, DB_PATH
from router import QueryPipeline

# ─── 설정 ──────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
COMMERCIAL_API_KEY = os.getenv("COMMERCIAL_API_KEY", "")
# 기본 모드: pipeline_only (로컬 LLM 없이 동작)
# COMMERCIAL_API_KEY가 설정되면 자동으로 상용 API 사용
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "pipeline_only")
FALLBACK_MODEL = "pipeline_only"
MAX_HISTORY = 10  # 대화 맥락 유지 최대 턴 수
COMMERCIAL_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _strip_thinking_tokens(text: str) -> str:
    """Qwen3 등 thinking 모델의 <think>...</think> 토큰 제거"""
    if not text:
        return text
    # <think>...</think> 블록 전체 제거 (multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ─── Commercial API 클라이언트 ─────────────────────────────────
class CommercialClient:
    """Commercial API를 통해 Claude 모델 사용 (로컬 LLM 불필요)"""

    def __init__(self, model: str = COMMERCIAL_DEFAULT_MODEL, api_key: str = COMMERCIAL_API_KEY):
        self.model = model
        self.api_key = api_key
        self.available = bool(api_key)
        if self.available:
            try:
                import anthropic  # Commercial API import -> anthropic 설치
            except ImportError:
                print("  ⚠️  Commercial 패키지 미설치. pip install anthropic")
                self.available = False

    def generate(self, system_prompt: str, messages: list,
                 stream: bool = True, options: dict = None) -> str:
        """Commercial API 호출"""
        if not self.available:
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            max_tokens = (options or {}).get("num_predict", 1024)

            if stream:
                full_response = ""
                with client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                ) as stream_obj:
                    for text in stream_obj.text_stream:
                        print(text, end="", flush=True)
                        full_response += text
                print()
                return full_response
            else:
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                )
                return resp.content[0].text
        except Exception as e:
            print(f"  ⚠️  Commercial API 오류: {e}")
            return None


# ─── Ollama LLM 클라이언트 ────────────────────────────────────
class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model
        self.available = self._check_connection()

    def get_available_models(self) -> list:
        """설치된 모델 목록 가져오기"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
            return []
        except Exception:
            return []

    def _check_connection(self) -> bool:
        if self.model == "pipeline_only":
            return False
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                if self.model in models:
                    return True
                # 사용 가능한 모델 중 아무거나 선택
                for m in models:
                    self.model = m
                    print(f"  ℹ️  사용 모델: {self.model}")
                    return True
                print(f"  ⚠️  Ollama 모델 없음. 사용 가능: {models}")
                return False
            return False
        except Exception:
            return False

    def generate(self, system_prompt: str, messages: list,
                 stream: bool = True, options: dict = None) -> str:
        """Ollama API 호출 (스트리밍 + thinking 토큰 필터링)"""
        if not self.available:
            return None

        gen_options = {"temperature": 0.3, "top_p": 0.9, "num_predict": 1024}
        if options:
            gen_options.update(options)

        # Qwen3 thinking 모드 비활성화 (qwen3:4b 등 지원)
        is_qwen3 = "qwen3" in self.model.lower()
        if is_qwen3:
            gen_options["think"] = False

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream": stream,
            "options": gen_options,
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=stream,
                timeout=300,
            )
            r.raise_for_status()

            if stream:
                full_response = ""
                in_think = False
                for line in r.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                # thinking 블록 실시간 필터링
                                if "<think>" in token:
                                    in_think = True
                                if not in_think:
                                    print(token, end="", flush=True)
                                    full_response += token
                                if "</think>" in token:
                                    in_think = False
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            pass
                print()
                return _strip_thinking_tokens(full_response)
            else:
                content = r.json().get("message", {}).get("content", "")
                return _strip_thinking_tokens(content)

        except Exception as e:
            print(f"  ⚠️  LLM 호출 오류: {e}")
            return None


# ─── 시스템 프롬프트 ───────────────────────────────────────────
def build_system_prompt(action_set_summary: str) -> str:
    """LLM 시스템 프롬프트: Action Set을 컨텍스트로 주입"""
    return f"""당신은 삼성전자 감사보고서(2014~2024년) 전문 금융 분석 AI 어시스턴트입니다.

## 역할
- 11년치 삼성전자 감사보고서 데이터를 기반으로 정확한 재무 정보를 제공합니다
- 데이터 기반 답변: 정형 DB에서 조회한 실제 수치를 사용합니다
- 분석적 통찰: 단순 수치 전달을 넘어 맥락과 의미를 설명합니다

## 사전 정의된 Action Set
{action_set_summary}

## 응답 원칙
1. **정확성 우선**: 조회된 실제 데이터만 사용, 추측하지 않습니다
2. **맥락 제공**: 수치의 의미와 변화 이유를 간략히 설명합니다
3. **범위 명시**: 수행 불가능한 질문은 그 이유와 대안을 안내합니다
4. **간결성**: 핵심 정보를 명확하고 간결하게 전달합니다
5. **한국어**: 모든 응답은 자연스러운 한국어로 작성합니다
6. **강조 표시**: 중요한 수치는 **볼드체**로 표시합니다
7. **중국어 사용금지**: 절대 중국어/일본어/한자를 사용하지 않습니다
8. **숫자 수치**: 숫자 수치는 반드시 파이프라인 답변의 원래 값과 단위(백만원)를 그대로 유지하세요.

## 데이터 범위
- 기간: 2014년 ~ 2024년 (11개년)
- 대상: 삼성전자주식회사 별도 재무제표
- 포함 정보: 재무상태표, 손익계산서, 현금흐름표, 감사의견"""


# ─── 대화 기록 관리 ───────────────────────────────────────────
class ConversationManager:
    def __init__(self, db_path: str = DB_PATH, session_id: str = None):
        self.db_path = db_path
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.history = []  # (role, content) 리스트

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # DB 저장
        conn = get_connection(self.db_path)
        conn.execute(
            "INSERT INTO conversation_history (session_id, role, content) VALUES (?, ?, ?)",
            (self.session_id, role, content)
        )
        conn.commit()
        conn.close()

    def get_messages(self, max_turns: int = MAX_HISTORY) -> list:
        """최근 N턴 대화 반환 (LLM 컨텍스트용)"""
        return self.history[-max_turns * 2:]

    def clear(self):
        self.history = []


# ─── 응답 포맷터 ──────────────────────────────────────────────
def format_pipeline_result(result: dict) -> str:
    """파이프라인 실행 결과 → 표시용 텍스트"""
    r = result.get("result", {})
    status = r.get("status", "unknown")

    if status == "success":
        return r.get("answer", "")
    elif status in ("infeasible", "exception", "no_data", "error"):
        return r.get("message", "처리할 수 없는 요청입니다.")
    return "처리 결과를 가져오지 못했습니다."


def enhance_with_llm(client, pipeline_answer: str,
                     query: str, conversation: ConversationManager,
                     system_prompt: str, options: dict = None) -> str:
    """파이프라인 결과 + LLM으로 응답 풍부화 (OllamaClient / CommercialClient 공통)"""

    messages = conversation.get_messages()
    augmented_query = f"""사용자 질문: {query}

[데이터 조회 결과]
{pipeline_answer}

위 데이터를 기반으로 사용자에게 자연스럽고 통찰력 있는 답변을 제공해 주세요.
- 이 시스템의 모든 재무 수치는 '백만원' 단위로 저장되어 있습니다.
예: 170,374,090백만원 = 약 170조 원
답변 시 파이프라인이 제공한 숫자와 '백만원' 단위를 그대로 사용하고,
절대 임의로 '원', '억원', '조원' 등으로 변환하지 마세요.
- 1~3문장으로 핵심을 설명하세요
- 관련 맥락이나 의미가 있다면 간략히 추가하세요
- 반드시 한국어로 답변하세요"""

    messages.append({"role": "user", "content": augmented_query})
    response = client.generate(system_prompt, messages, options=options)
    return response


def _create_llm_client():
    """LLM 클라이언트 자동 선택:
    1) Commercial API (API 키 있으면)
    2) Ollama 로컬 (연결 가능하면)
    3) Pipeline-only (fallback)
    """
    # 1순위: Commercial API
    if COMMERCIAL_API_KEY:
        client = CommercialClient()
        if client.available:
            print(f"  ✅ LLM: Commercial API ({COMMERCIAL_DEFAULT_MODEL})")
            return client

    # 2순위: Ollama 로컬
    ollama_model = os.getenv("OLLAMA_MODEL", "")
    if not ollama_model or ollama_model == "pipeline_only":
        # 연결 가능한지만 확인 (모델은 select_model()에서 선택)
        ollama_model = "auto"
    client = OllamaClient(model=ollama_model)
    if client.available:
        print(f"  ✅ LLM: Ollama ({client.model})")
        return client

    # 3순위: Pipeline-only
    print("  ℹ️  LLM: 파이프라인 단독 모드 (로컬 모델 미사용)")
    return None


# ─── 메인 챗봇 ────────────────────────────────────────────────
class AuditReportChatbot:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.pipeline = QueryPipeline(db_path)
        self.llm = _create_llm_client()  # None이면 pipeline-only
        self.action_set = self.pipeline.get_action_set_summary()
        self.system_prompt = build_system_prompt(self.action_set)
        self.conversation = ConversationManager(db_path)

        # Ollama인 경우에만 모델 선택 UI 표시
        if isinstance(self.llm, OllamaClient):
            self.select_model()

    def _llm_mode_label(self) -> str:
        if self.llm is None:
            return "파이프라인 단독 모드"
        if isinstance(self.llm, CommercialClient):
            return f"Commercial API ({self.llm.model})"
        return f"Ollama ({self.llm.model})"

    def select_model(self):
        ollama_models = []
        if isinstance(self.llm, OllamaClient) and self.llm.available:
            ollama_models = sorted(self.llm.get_available_models())

        print("\n" + "="*45)
        print("  LLM 모드 선택:")
        print("="*45)

        options = ["pipeline_only (LLM 없이 동작)"] + ollama_models
        current_label = self._llm_mode_label()
        for i, m in enumerate(options, 1):
            tag = " [현재]" if (
                (i == 1 and self.llm is None) or
                (isinstance(self.llm, OllamaClient) and m == self.llm.model)
            ) else ""
            print(f"  [{i}] {m}{tag}")
        print("="*45)

        try:
            choice = input(f"\n  번호 선택 (현재: {current_label}, 엔터=유지): ").strip()
            if choice and choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    if idx == 0:
                        self.llm = None
                        print("  ✅ pipeline_only 모드로 전환되었습니다.")
                    else:
                        model_name = ollama_models[idx - 1]
                        if not isinstance(self.llm, OllamaClient):
                            self.llm = OllamaClient(model=model_name)
                        else:
                            self.llm.model = model_name
                        print(f"  ✅ 모델: '{model_name}'")
                else:
                    print(f"  ℹ️  유지: {current_label}")
            else:
                print(f"  ℹ️  유지: {current_label}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n  ℹ️  유지: {current_label}")

    def chat(self, user_input: str, verbose: bool = False) -> str:
        """단일 턴 처리"""
        # 1. 파이프라인 실행
        pipeline_result = self.pipeline.process(user_input, verbose=verbose)
        pipeline_answer = format_pipeline_result(pipeline_result)

        if verbose:
            print(f"  📌 Task: {pipeline_result['task_name']} [{pipeline_result['task_group']}]")

        # 2. LLM 응답 풍부화 (LLM 클라이언트가 있는 경우)
        if self.llm is not None:
            print("  🤖 ", end="", flush=True)
            enhanced = enhance_with_llm(
                self.llm, pipeline_answer, user_input,
                self.conversation, self.system_prompt,
            )
            final_answer = enhanced if enhanced else pipeline_answer
        else:
            final_answer = pipeline_answer

        # 3. 대화 기록 저장
        self.conversation.add("user", user_input)
        self.conversation.add("assistant", final_answer)

        return final_answer

    def print_welcome(self):
        """시작 메시지"""
        print("\n" + "="*65)
        print("  📊 삼성전자 감사보고서 QA 챗봇 (2014~2024)")
        print("="*65)
        print(f"  LLM 모드: {self._llm_mode_label()}")
        print(f"  세션 ID: {self.conversation.session_id}")
        print("─"*65)
        print("  예시 질문:")
        print("  • 2023년 매출액은?")
        print("  • 2019년과 2023년 영업이익 비교해줘")
        print("  • 2015년 대비 2016년 영업이익 증감액은?")
        print("  • 최근 10년 자산총계 추이 보여줘")
        print("  • 2023년 부채비율 계산해줘")
        print("  • 2022년 감사의견은?")
        print("  • 2017년 주석 19는 어떤 주제를 다루는가?")
        print("─"*65)
        print("  명령어: /help | /clear | /data | /quit | /model")
        print("="*65 + "\n")

    def handle_command(self, cmd: str) -> tuple:
        """슬래시 명령어 처리 → (handled, response)"""
        cmd = cmd.strip().lower()

        if cmd in ("/quit", "/exit"):
            print("\n  챗봇을 종료합니다. 감사합니다! 👋")
            return True, None

        if cmd == "/clear":
            self.conversation.clear()
            return True, "✅ 대화 기록을 초기화했습니다."

        if cmd == "/model":
            self.select_model()
            return True, f"현재 모드: {self._llm_mode_label()}"

        if cmd == "/help":
            return True, (
                "도움말:\n"
                "• /data  - 보유 데이터 목록\n"
                "• /model - LLM 모델 변경\n"
                "• /clear - 대화 기록 초기화\n"
                "• /quit  - 종료\n\n"
                "가능한 질문 유형:\n"
                "• 특정 연도 재무 지표 조회\n"
                "• 두 연도 간 비교\n"
                "• 10년 추이 분석\n"
                "• 재무비율 계산 (부채비율, ROE, ROA 등)\n"
                "• 감사의견 조회\n"
                "• 키워드 검색"
            )

        if cmd == "/data":
            from tools import list_available_data
            data = list_available_data(self.db_path)
            return True, (
                f"보유 데이터:\n"
                f"• 연도: {', '.join(map(str, data['years']))}\n"
                f"• 재무 지표: {', '.join(data['metrics'])}"
            )

        return False, None

    def run(self):
        """대화형 CLI 실행"""
        self.print_welcome()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  종료합니다.")
                break

            if not user_input:
                continue

            # 슬래시 명령어
            if user_input.startswith("/"):
                handled, response = self.handle_command(user_input)
                if handled:
                    if response is None:
                        break
                    print(f"\nBot: {response}\n")
                    continue

            # 일반 질의 처리
            print(f"\nBot: ", end="", flush=True)
            answer = self.chat(user_input)

            # LLM 스트리밍이 아닌 경우에만 출력
            if self.llm is None or not self.llm.available:
                print(answer)

            print()  # 빈 줄


# ─── 비대화형 테스트 모드 ─────────────────────────────────────
def run_batch_test(db_path: str = DB_PATH):
    """자동 테스트: 다양한 질의 검증"""
    print("\n" + "="*65)
    print("  🧪 자동 테스트 모드")
    print("="*65)

    pipeline = QueryPipeline(db_path)

    test_cases = [
        # (질의, 기대 task_group, 설명)
        ("2023년 매출액은 얼마야?", "feasible", "단일연도 지표"),
        ("2019년과 2023년 영업이익 비교해줘", "feasible", "연도비교"),
        ("최근 10년 자산총계 추이", "feasible", "추이분석"),
        ("영업이익이 가장 높았던 해는?", "feasible", "극값탐색"),
        ("2023년 감사의견은?", "feasible", "감사의견"),
        ("2023년 부채비율 계산해줘", "feasible", "재무비율"),
        ("2022년 종합 요약해줘", "feasible", "연도요약"),
        ("어떤 데이터 조회 가능해?", "feasible", "데이터목록"),
        ("삼성전자 주식 사도 돼?", "infeasible", "투자추천"),
        ("2025년 매출 예측해줘", "infeasible", "미래예측"),
        ("SK하이닉스와 비교해줘", "infeasible", "경쟁사비교"),
        ("매출액 알려줘", "exception", "연도미지정"),
    ]

    passed = 0
    for query, expected_group, desc in test_cases:
        result = pipeline.process(query)
        actual_group = result["task_group"]
        status = result["result"].get("status", "")
        answer = result["result"].get("answer", result["result"].get("message", ""))

        ok = "✅" if actual_group == expected_group else "❌"
        if actual_group == expected_group:
            passed += 1

        print(f"\n{ok} [{desc}]")
        print(f"   Q: {query}")
        print(f"   Task: {result['task_name']} [{actual_group}] (신뢰도: {result.get('confidence', 0):.3f})")
        print(f"   답변: {answer[:100]}...")

    print(f"\n{'='*65}")
    print(f"  테스트 결과: {passed}/{len(test_cases)} 통과")
    print(f"{'='*65}")
    return passed, len(test_cases)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="삼성전자 감사보고서 QA 챗봇")
    parser.add_argument("--test", action="store_true", help="자동 테스트 실행")
    parser.add_argument("--db", default=DB_PATH, help="DB 경로")
    parser.add_argument("--verbose", action="store_true", help="상세 출력")
    args = parser.parse_args()

    if args.test:
        run_batch_test(args.db)
    else:
        bot = AuditReportChatbot(args.db)
        bot.run()
