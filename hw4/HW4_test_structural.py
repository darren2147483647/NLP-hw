"""
test_structural.py - Prompt-based × Structural-Only 推論腳本（多執行緒版）

功能：讀取 test.jsonl → 多執行緒並行呼叫 Gemma-4 API → 輸出 submission.csv
    每把 API Key 一條專屬執行緒，同時發送請求以最大化吞吐量。
    所有 API call 記錄於 logs/test_structural_log.jsonl

使用方式:
  python test_structural.py
"""

import json
import os
import sys
import time
import threading
from queue import Queue, Empty
from collections import Counter

import pandas as pd

from utils import (
    load_api_keys,
    load_jsonl,
    init_log,
    append_log,
    query_gemma,
    extract_answer,
    heuristic_fallback,
    RateLimiter,
    get_prompt_builder,
    now_iso,
)


# ============================================================
# Worker: 每條執行緒的工作邏輯
# ============================================================
THREADS_PER_KEY = 6  # 每把 API key 分配的執行緒數量


def _worker(
    worker_id,
    key_idx,
    api_key,
    system_prompt,
    build_prompt_fn,
    task_queue,
    results,
    results_lock,
    log_path,
    log_lock,
    counter,
    counter_lock,
    start_idx,
    total_count,
    stop_event,
    shared_rate_limiter,
):
    """
    Worker 執行緒：從 task_queue 取任務、呼叫 API、存結果。
    同一把 key 的多條 thread 共用一個 RateLimiter。
    """
    rate_limiter = shared_rate_limiter

    while not stop_event.is_set():
        try:
            i, sample = task_queue.get(timeout=2)
        except Empty:
            break  # 佇列已空，結束

        sample_id = sample["id"]
        user_prompt, valid_options = build_prompt_fn(sample)

        # Per-key rate limiting
        rate_limiter.wait()

        # 呼叫 API（額外 try/except 以防未預期的例外）
        try:
            response_text, retries, error_msg = query_gemma(
                system_prompt, user_prompt, api_key
            )
        except Exception as e:
            response_text = None
            retries = 0
            error_msg = f"未捕獲例外: {str(e)}"

        rate_limiter.tick()

        # 擷取答案
        if response_text is not None:
            answer, confidence = extract_answer(response_text, valid_options)
            if answer is None:
                answer = heuristic_fallback(sample, valid_options)
                confidence = "heuristic"
        else:
            answer = heuristic_fallback(sample, valid_options)
            confidence = "api_failed"

        # 寫入 log（thread-safe）
        log_record = {
            "timestamp": now_iso(),
            "id": sample_id,
            "config": "structural",
            "user_prompt": user_prompt,
            "raw_response": response_text,
            "extracted_answer": answer,
            "confidence": confidence,
            "retries": retries,
            "error": error_msg,
        }
        with log_lock:
            append_log(log_path, log_record)

        # 儲存結果（thread-safe）
        with results_lock:
            results[i] = {"id": sample_id, "answer": answer}

        # 計數 + 顯示進度
        with counter_lock:
            counter[0] += 1
            done = counter[0]

        print(
            f"  [{start_idx + done}/{total_count}] ID={sample_id} → {answer} "
            f"(conf={confidence}, retries={retries}, key={key_idx})",
            flush=True,
        )

        task_queue.task_done()


# ============================================================
# Progress helpers
# ============================================================
def _collect_contiguous_results(base_data, results, start_idx, total_count):
    """
    從 results dict 中收集從 start_idx 開始的連續結果。
    遇到第一個缺口就停止（確保斷點續傳的正確性）。
    """
    ordered = list(base_data)  # 先複製已有的進度
    for idx in range(start_idx, total_count):
        if idx in results:
            ordered.append(results[idx])
        else:
            break  # 遇到缺口即停止
    return ordered


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
    print(f"✅ [Structural-Only] 處理完成！共 {total_count} 筆測試資料")
    print(f"📄 Kaggle 上傳檔: submission.csv")
    print(f"💾 API 呼叫記錄: logs/test_structural_log.jsonl")
    print("=" * 60)

    # 答案分布
    answer_counts = Counter(d["answer"] for d in submission_data)
    print("\n📊 答案分布:")
    for letter in sorted(answer_counts.keys()):
        count = answer_counts[letter]
        pct = count / len(submission_data) * 100
        bar = "█" * (count // 5)
        print(f"   {letter}: {count:4d} ({pct:5.1f}%) {bar}")


# ============================================================
# Main
# ============================================================
def main():
    api_keys = load_api_keys()
    num_keys = len(api_keys)
    total_threads = num_keys * THREADS_PER_KEY
    print(f"🔑 載入 {num_keys} 把 API Key × {THREADS_PER_KEY} threads = {total_threads} 條並行")

    # 使用 Structural-Only 配置
    system_prompt, build_prompt_fn = get_prompt_builder("structural")

    # ---- 載入資料 ----
    print("📂 載入 test.jsonl 資料...")
    test_data = load_jsonl("test.jsonl")
    total_count = len(test_data)
    print(f"   共 {total_count} 筆測試資料")
    print(f"🔧 Prompt 配置: Structural-Only (多執行緒)")

    # ---- 初始化 log ----
    log_path = init_log("logs/test_structural_log.jsonl")
    log_lock = threading.Lock()
    print(f"📝 API 呼叫記錄將寫入: {log_path}")

    # ---- 斷點續傳：檢查進度 ----
    progress_file = "logs/test_structural_progress.json"
    base_submission_data = []
    start_idx = 0

    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
            base_submission_data = saved.get("submission_data", [])
            start_idx = len(base_submission_data)
            if start_idx > 0 and start_idx < total_count:
                print(f"🔄 找到進度檔，從第 {start_idx + 1} 筆繼續")
            elif start_idx >= total_count:
                print(f"✅ 進度檔顯示已全部完成 ({start_idx}/{total_count})")
                print("   若要重跑，請刪除 logs/test_structural_progress.json")
                _save_submission(base_submission_data, total_count)
                return

    remaining = total_count - start_idx
    print(f"🚀 剩餘 {remaining} 筆，使用 {total_threads} 條執行緒並行處理")

    # ---- 建立任務佇列 ----
    task_queue = Queue()
    for i in range(start_idx, total_count):
        task_queue.put((i, test_data[i]))

    # ---- Thread-safe 共用狀態 ----
    results = {}              # {global_index: {"id": ..., "answer": ...}}
    results_lock = threading.Lock()
    counter = [0]             # mutable int for thread access
    counter_lock = threading.Lock()
    stop_event = threading.Event()

    # ---- 每把 key 建一個共用的 RateLimiter ----
    per_key_limiters = [RateLimiter() for _ in range(num_keys)]

    # ---- 啟動 worker 執行緒（每把 key × THREADS_PER_KEY 條）----
    threads = []
    worker_id = 0
    for k in range(num_keys):
        for t_idx in range(THREADS_PER_KEY):
            t = threading.Thread(
                target=_worker,
                args=(
                    worker_id,
                    k, api_keys[k],
                    system_prompt, build_prompt_fn,
                    task_queue,
                    results, results_lock,
                    log_path, log_lock,
                    counter, counter_lock,
                    start_idx, total_count,
                    stop_event,
                    per_key_limiters[k],
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)
            worker_id += 1

    # ---- 主執行緒：等待完成 + 定期儲存進度 ----
    last_saved_count = 0
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(10)  # 每 10 秒檢查一次
            with results_lock:
                current_results = dict(results)

            contiguous = _collect_contiguous_results(
                base_submission_data, current_results, start_idx, total_count
            )
            if len(contiguous) > last_saved_count:
                _save_progress(contiguous, progress_file)
                last_saved_count = len(contiguous)
                print(
                    f"  💾 進度已儲存: {last_saved_count}/{total_count}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n⚠️  收到中斷信號，正在儲存進度...")
        stop_event.set()
        # 等待 worker 結束（最多 5 秒）
        for t in threads:
            t.join(timeout=5)

    # ---- 最終儲存 ----
    with results_lock:
        current_results = dict(results)

    final_data = _collect_contiguous_results(
        base_submission_data, current_results, start_idx, total_count
    )
    _save_progress(final_data, progress_file)

    total_done = len(current_results)
    print(f"\n💾 最終進度已儲存: {len(final_data)}/{total_count} (本次完成 {total_done} 筆)")

    if len(final_data) >= total_count:
        _save_submission(final_data, total_count)
    else:
        print(f"⚠️  尚有 {total_count - len(final_data)} 筆未完成，重跑即可繼續")


if __name__ == "__main__":
    main()
