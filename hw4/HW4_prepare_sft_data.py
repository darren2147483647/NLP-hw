"""
prepare_sft_data.py - SFT 訓練資料準備腳本

功能：將 train.jsonl 轉換成 SFT 訓練格式（chat messages format）
    分別產生 Full-Information 和 Structural-Only 兩種版本的資料集

使用方式:
  python prepare_sft_data.py
  python prepare_sft_data.py --val_ratio 0.1 --seed 42

輸出:
  sft_data/full_info_train.jsonl
  sft_data/full_info_val.jsonl
  sft_data/structural_train.jsonl
  sft_data/structural_val.jsonl
"""

import json
import os
import sys
import random
import argparse

from utils import (
    load_jsonl,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_STRUCTURAL,
    build_tool_calling_prompt,
    build_structural_only_prompt,
)


def sample_to_sft_format(sample, system_prompt, build_fn):
    """
    將一筆 sample 轉換成 SFT chat format。

    回傳:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "Answer: X"}
      ]
    }
    """
    user_prompt, _ = build_fn(sample)
    answer = sample["answer"]

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f"Answer: {answer}"},
        ]
    }


def save_jsonl(data, filepath):
    """儲存 list of dict 為 JSONL"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"   ✅ 已儲存 {len(data)} 筆 → {filepath}")


def main():
    parser = argparse.ArgumentParser(description="準備 SFT 訓練資料")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="驗證集比例 (預設 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機種子 (預設 42)")
    parser.add_argument("--data_path", type=str, default="train.jsonl",
                        help="訓練資料路徑")
    parser.add_argument("--output_dir", type=str, default="sft_data",
                        help="輸出目錄")
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- 載入資料 ----
    print("📂 載入訓練資料...")
    all_data = load_jsonl(args.data_path)
    print(f"   共 {len(all_data)} 筆")

    # ---- 隨機拆分 train/val ----
    random.shuffle(all_data)
    val_size = int(len(all_data) * args.val_ratio)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    print(f"   Train: {len(train_data)} 筆, Val: {len(val_data)} 筆")

    # ---- 轉換 Full-Information 版本 ----
    print("\n🔧 轉換 Full-Information 資料...")
    full_train = [
        sample_to_sft_format(s, SYSTEM_PROMPT, build_tool_calling_prompt)
        for s in train_data
    ]
    full_val = [
        sample_to_sft_format(s, SYSTEM_PROMPT, build_tool_calling_prompt)
        for s in val_data
    ]
    save_jsonl(full_train, os.path.join(args.output_dir, "full_info_train.jsonl"))
    save_jsonl(full_val, os.path.join(args.output_dir, "full_info_val.jsonl"))

    # ---- 轉換 Structural-Only 版本 ----
    print("\n🔧 轉換 Structural-Only 資料...")
    struct_train = [
        sample_to_sft_format(s, SYSTEM_PROMPT_STRUCTURAL, build_structural_only_prompt)
        for s in train_data
    ]
    struct_val = [
        sample_to_sft_format(s, SYSTEM_PROMPT_STRUCTURAL, build_structural_only_prompt)
        for s in val_data
    ]
    save_jsonl(struct_train, os.path.join(args.output_dir, "structural_train.jsonl"))
    save_jsonl(struct_val, os.path.join(args.output_dir, "structural_val.jsonl"))

    # ---- 統計 ----
    print("\n" + "=" * 60)
    print("📊 資料準備完成")
    print("=" * 60)

    # 答案分布
    from collections import Counter
    answer_counts = Counter(s["answer"] for s in all_data)
    print("\n📈 答案分布 (全部資料):")
    for letter in sorted(answer_counts.keys()):
        count = answer_counts[letter]
        pct = count / len(all_data) * 100
        print(f"   {letter}: {count:5d} ({pct:5.1f}%)")

    # Prompt 長度統計
    sample_prompts = [
        build_tool_calling_prompt(s)[0] for s in all_data[:100]
    ]
    avg_len = sum(len(p) for p in sample_prompts) / len(sample_prompts)
    max_len = max(len(p) for p in sample_prompts)
    print(f"\n📏 Full-Info Prompt 長度 (前 100 筆):")
    print(f"   平均: {avg_len:.0f} chars, 最大: {max_len} chars")

    sample_prompts_s = [
        build_structural_only_prompt(s)[0] for s in all_data[:100]
    ]
    avg_len_s = sum(len(p) for p in sample_prompts_s) / len(sample_prompts_s)
    max_len_s = max(len(p) for p in sample_prompts_s)
    print(f"\n📏 Structural-Only Prompt 長度 (前 100 筆):")
    print(f"   平均: {avg_len_s:.0f} chars, 最大: {max_len_s} chars")

    print(f"\n💾 輸出目錄: {args.output_dir}/")
    print(f"   full_info_train.jsonl   ({len(full_train)} 筆)")
    print(f"   full_info_val.jsonl     ({len(full_val)} 筆)")
    print(f"   structural_train.jsonl  ({len(struct_train)} 筆)")
    print(f"   structural_val.jsonl    ({len(struct_val)} 筆)")


if __name__ == "__main__":
    main()
