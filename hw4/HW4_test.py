"""
test.py - HW4 Tool Calling Agent 推論腳本

功能：讀取 test.jsonl → 逐筆呼叫 Gemma-4 API → 輸出 submission.csv
所有 API call 記錄於 logs/test_log.jsonl

使用方式:
  python test.py
"""

import json
import os
import sys
import pandas as pd
from collections import Counter

from utils import (
    load_api_key,
    load_jsonl,
    init_log,
    append_log,
    query_gemma,
    build_tool_calling_prompt,
    extract_answer,
    heuristic_fallback,
    RateLimiter,
    SYSTEM_PROMPT,
    now_iso,
)


def main():
    api_key = load_api_key()
    rate_limiter = RateLimiter()

    # ---- 載入資料 ----
    print("📂 載入 test.jsonl 資料...")
    test_data = load_jsonl("test.jsonl")
    total_count = len(test_data)
    print(f"   共 {total_count} 筆測試資料")

    # ---- 初始化 log ----
    log_path = init_log("logs/test_log.jsonl")
    print(f"📝 API 呼叫記錄將寫入: {log_path}")

    # ---- 斷點續傳：檢查進度 ----
    progress_file = "logs/test_progress.json"
    submission_data = []
    start_idx = 0

    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
            submission_data = saved.get("submission_data", [])
            start_idx = len(submission_data)
            if start_idx > 0 and start_idx < total_count:
                print(f"🔄 找到進度檔，從第 {start_idx + 1} 筆繼續")
            elif start_idx >= total_count:
                print(f"✅ 進度檔顯示已全部完成 ({start_idx}/{total_count})")
                print("   若要重跑，請刪除 logs/test_progress.json")
                _save_submission(submission_data, total_count)
                return

    # ---- 逐筆推論 ----
    for i in range(start_idx, total_count):
        sample = test_data[i]
        sample_id = sample["id"]

        user_prompt, valid_options = build_tool_calling_prompt(sample)

        # Rate limiting
        rate_limiter.wait()

        # 呼叫 API
        response_text, retries, error_msg = query_gemma(
            SYSTEM_PROMPT, user_prompt, api_key
        )
        rate_limiter.tick()

        # 擷取答案
        if response_text is not None:
            answer, confidence = extract_answer(response_text, valid_options)
            if answer is None:
                # 模型有回應但無法擷取 → 啟發式
                answer = heuristic_fallback(sample, valid_options)
                confidence = "heuristic"
        else:
            # API 完全失敗 → 啟發式
            answer = heuristic_fallback(sample, valid_options)
            confidence = "api_failed"

        # 記錄
        log_record = {
            "timestamp": now_iso(),
            "id": sample_id,
            "user_prompt": user_prompt,
            "raw_response": response_text,
            "extracted_answer": answer,
            "confidence": confidence,
            "retries": retries,
            "error": error_msg,
        }
        append_log(log_path, log_record)

        submission_data.append({"id": sample_id, "answer": answer})

        # 進度顯示
        if i == start_idx or (i + 1) % 10 == 0 or (i + 1) == total_count:
            print(
                f"  [{i+1}/{total_count}] ID={sample_id} → {answer} "
                f"(conf={confidence}, retries={retries})",
                flush=True,
            )

        # 每 50 筆存一次進度
        if (i + 1) % 50 == 0:
            _save_progress(submission_data, progress_file)

    # ---- 輸出 ----
    _save_progress(submission_data, progress_file)
    _save_submission(submission_data, total_count)


def _save_progress(submission_data, progress_file):
    """儲存斷點續傳進度"""
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump({"submission_data": submission_data}, f, ensure_ascii=False)


def _save_submission(submission_data, total_count):
    """儲存 submission.csv"""
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv("submission.csv", index=False)

    print("\n" + "=" * 60)
    print(f"✅ 處理完成！共 {total_count} 筆測試資料")
    print(f"📄 Kaggle 上傳檔: submission.csv")
    print(f"💾 API 呼叫記錄: logs/test_log.jsonl")
    print("=" * 60)

    # 答案分布
    answer_counts = Counter(d["answer"] for d in submission_data)
    print("\n📊 答案分布:")
    for letter in sorted(answer_counts.keys()):
        count = answer_counts[letter]
        pct = count / len(submission_data) * 100
        bar = "█" * (count // 5)
        print(f"   {letter}: {count:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
