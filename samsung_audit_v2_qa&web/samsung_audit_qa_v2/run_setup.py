"""
run_setup.py - 원클릭 전체 파이프라인 실행
1. DB 초기화 + Task Catalog 등록
2. HTML 파싱 + 데이터 적재
3. 챗봇 실행
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))
from db_schema import setup_database, DB_PATH
from ingest import run_ingestion
from chatbot import run_batch_test, AuditReportChatbot


def find_data_dir():
    """감사보고서 HTML 파일 디렉토리 자동 탐색 (한글 경로 지원)"""
    # glob 기반으로 한글 경로에서도 탐색 가능
    search_patterns = [
        "/sessions/**/감사보고서_2014.htm",
        os.path.join(os.path.expanduser("~"), "**/감사보고서_2014.htm"),
        "./**/감사보고서_2014.htm",
    ]
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return os.path.dirname(matches[0])

    # fallback: 직접 경로
    candidates = [
        "/sessions/dazzling-focused-hawking/mnt/클로드전용/삼성전자_감사보고서_2014_2024",
    ]
    for path in candidates:
        try:
            files = glob.glob(os.path.join(path, "*.htm"))
            if files:
                return path
        except Exception:
            pass

    return None


def main():
    print("=" * 65)
    print("  🚀 삼성전자 감사보고서 NLP 시스템 초기화")
    print("=" * 65)

    # ─ Step 1: 데이터 디렉토리 탐색
    data_dir = find_data_dir()
    if not data_dir:
        print("❌ 감사보고서 HTML 파일을 찾을 수 없습니다.")
        print("   사용법: python run_setup.py [데이터폴더경로]")
        sys.exit(1)

    print(f"\n✅ 데이터 경로: {data_dir}")

    # ─ Step 2: DB 초기화 + Task Catalog 등록
    print("\n[STEP 1/3] DB 초기화 및 Action Set 등록...")
    setup_database(DB_PATH)

    # ─ Step 3: 파싱 + 적재
    print("\n[STEP 2/3] HTML 파싱 및 DB 적재...")
    stats = run_ingestion(data_dir, DB_PATH)

    # ─ Step 4: 자동 테스트
    print("\n[STEP 3/3] 시스템 테스트...")
    passed, total = run_batch_test(DB_PATH)

    print(f"\n{'='*65}")
    print(f"  ✅ 시스템 초기화 완료!")
    print(f"     DB: {DB_PATH}")
    print(f"     테스트: {passed}/{total} 통과")
    print(f"\n  챗봇 실행:")
    print(f"     python chatbot.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 명시적 데이터 경로
        import importlib
        from db_schema import setup_database
        setup_database(DB_PATH)

        from ingest import run_ingestion
        run_ingestion(sys.argv[1], DB_PATH)

        run_batch_test(DB_PATH)
    else:
        main()
