"""
eval.py - HW4 Tool Calling Agent 驗證腳本

功能：從 train.jsonl 隨機抽樣 N 筆，呼叫 API 做預測並計算準確率
所有 API call 記錄於 logs/eval_log.jsonl

使用方式:
  python eval.py              # 預設抽樣 50 筆
  python eval.py 100          # 抽樣 100 筆
  python eval.py all          # 全部 13587 筆（需要很久）
"""

import json
import os
import sys
import random
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

    # ---- 解析參數 ----
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower() == "all":
            num_samples = None  # 全部
        else:
            num_samples = int(arg)
    else:
        num_samples = 50

    # ---- 載入資料 ----
    print("📂 載入 train.jsonl 資料...")
    all_data = load_jsonl("train.jsonl")
    total_available = len(all_data)
    print(f"   共 {total_available} 筆訓練資料")

    if num_samples is None or num_samples >= total_available:
        samples = all_data
        num_samples = total_available
        print(f"   使用全部 {num_samples} 筆進行驗證")
    else:
        samples = random.sample(all_data, num_samples)
        print(f"   隨機抽樣 {num_samples} 筆進行驗證")

    # ---- 初始化 log ----
    log_path = init_log("logs/eval_log.jsonl")
    print(f"📝 API 呼叫記錄將寫入: {log_path}")

    # ---- 逐筆驗證 ----
    correct = 0
    total = 0
    errors = []
    confidence_stats = Counter()
    api_failures = 0

    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        ground_truth = sample["answer"]

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
                answer = heuristic_fallback(sample, valid_options)
                confidence = "heuristic"
        else:
            answer = heuristic_fallback(sample, valid_options)
            confidence = "api_failed"
            api_failures += 1

        is_correct = answer == ground_truth
        if is_correct:
            correct += 1
        else:
            errors.append({
                "id": sample_id,
                "predicted": answer,
                "ground_truth": ground_truth,
                "confidence": confidence,
                "response_preview": (response_text[:200] if response_text else "N/A"),
            })

        total += 1
        confidence_stats[confidence] += 1

        # 記錄
        log_record = {
            "timestamp": now_iso(),
            "id": sample_id,
            "user_prompt": user_prompt,
            "raw_response": response_text,
            "extracted_answer": answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "confidence": confidence,
            "retries": retries,
            "error": error_msg,
        }
        append_log(log_path, log_record)

        # 進度顯示
        accuracy = correct / total * 100
        status = "✅" if is_correct else "❌"
        print(
            f"  [{i+1}/{num_samples}] ID={sample_id:5d} | "
            f"pred={answer} gt={ground_truth} {status} | "
            f"Acc: {accuracy:.1f}% | conf={confidence}",
            flush=True,
        )

    # ---- 結果報告 ----
    print("\n" + "=" * 60)
    print(f"📊 驗證結果: {correct}/{total} = {correct/total*100:.2f}% 準確率")
    print("=" * 60)

    print(f"\n📈 信心度分布:")
    for conf_level in ["high", "medium", "low", "heuristic", "api_failed"]:
        count = confidence_stats.get(conf_level, 0)
        if count > 0:
            pct = count / total * 100
            print(f"   {conf_level:12s}: {count:4d} ({pct:5.1f}%)")

    if api_failures > 0:
        print(f"\n⚠️  API 失敗次數: {api_failures}")

    if errors:
        print(f"\n❌ 錯誤案例 (共 {len(errors)} 筆，顯示前 15 筆):")
        for e in errors[:15]:
            print(
                f"   ID={e['id']:5d}: pred={e['predicted']} "
                f"gt={e['ground_truth']} (conf={e['confidence']})"
            )
            if e["response_preview"] != "N/A":
                preview = e["response_preview"].replace("\n", " ")[:100]
                print(f"           resp: {preview}...")

    print(f"\n💾 完整 API 呼叫記錄: {log_path}")


if __name__ == "__main__":
    main()
