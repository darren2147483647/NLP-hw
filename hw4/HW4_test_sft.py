"""
test_sft.py - SFT 微調模型推論腳本 (Kaggle 提交用)

功能：載入微調後的 LoRA 權重 (.pt)，對 test.jsonl 做推論，輸出 submission.csv
    所有推論記錄於 logs/test_sft_log.jsonl

使用方式:
  python test_sft.py --model models/sft_full_info --config full_info
  python test_sft.py --model models/sft_structural --config structural
  python test_sft.py --model models/sft_full_info/lora_weights.pt --config full_info

全部參數:
  --model           微調後模型路徑（目錄或 .pt 檔案）(必填)
  --base_model      基底模型名稱 (預設: Qwen/Qwen2.5-7B-Instruct，會從 .pt 自動偵測)
  --config          full_info | structural (預設: full_info)
  --data_path       測試資料路徑 (預設: test.jsonl)
  --output          輸出 CSV 路徑 (預設: submission.csv)
  --max_new_tokens  生成最大 token 數 (預設: 64)
  --use_4bit        使用 4-bit 量化 (預設: True)
  --no_4bit         不使用 4-bit 量化
"""

import json
import os
import sys
import argparse
import warnings
import torch
import pandas as pd
from collections import Counter

from utils import (
    load_jsonl,
    init_log,
    append_log,
    extract_answer,
    heuristic_fallback,
    get_prompt_builder,
    now_iso,
)

# 抑制洗版 warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT 模型推論 (Kaggle 提交)")
    parser.add_argument("--model", type=str, default="models/sft_full_info",
                        help="微調後模型路徑（目錄或 .pt 檔案）")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="基底模型名稱")
    parser.add_argument("--config", type=str, default="full_info",
                        choices=["full_info", "structural"],
                        help="Prompt 配置")
    parser.add_argument("--data_path", type=str, default="test.jsonl",
                        help="測試資料路徑")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="輸出 CSV 路徑")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="生成最大 token 數")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="4-bit 量化載入基底模型")
    parser.add_argument("--no_4bit", action="store_true",
                        help="不使用 4-bit 量化")
    return parser.parse_args()


def load_sft_model(base_model_name, adapter_path, use_4bit=True):
    """
    載入基底模型 + LoRA adapter（從 .pt 檔）。

    adapter_path 可以是：
      - 目錄路徑（會自動尋找內部的 lora_weights.pt）
      - 直接指向 .pt 檔案
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    # 解析 .pt 路徑
    if os.path.isdir(adapter_path):
        pt_path = os.path.join(adapter_path, "lora_weights.pt")
    else:
        pt_path = adapter_path

    if not os.path.exists(pt_path):
        print(f"❌ 找不到權重檔: {pt_path}")
        sys.exit(1)

    # 載入 .pt
    print(f"📦 載入 .pt 權重檔: {pt_path}")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    lora_state_dict = checkpoint["lora_state_dict"]
    lora_config_dict = checkpoint["lora_config"]
    train_config = checkpoint.get("train_config", {})

    # 自動偵測基底模型
    saved_base = train_config.get("base_model")
    if saved_base:
        print(f"📋 從 .pt 自動偵測基底模型: {saved_base}")
        base_model_name = saved_base

    # 載入基底模型
    print(f"📦 載入基底模型: {base_model_name}")
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16 if not use_4bit else None,
    )

    # 重建 LoRA 配置並套用
    print(f"🔧 套用 LoRA adapter (rank={lora_config_dict['r']}, alpha={lora_config_dict['lora_alpha']})")
    lora_config = LoraConfig(**lora_config_dict)
    model = get_peft_model(model, lora_config)

    # 載入 LoRA 權重
    load_result = model.load_state_dict(lora_state_dict, strict=False)
    if load_result.unexpected_keys:
        print(f"⚠️  Unexpected keys: {load_result.unexpected_keys[:5]}...")
    print(f"   ✅ 已載入 {len(lora_state_dict)} 個 LoRA 參數")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_answer(model, tokenizer, system_prompt, user_prompt, max_new_tokens=64):
    """使用本地模型生成回答"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response


def main():
    args = parse_args()

    if args.no_4bit:
        args.use_4bit = False

    print("=" * 60)
    print("🚀 SFT 模型推論 (Kaggle 提交)")
    print("=" * 60)
    print(f"   模型: {args.model}")
    print(f"   配置: {args.config}")
    print(f"   測試資料: {args.data_path}")
    print(f"   輸出: {args.output}")
    print("=" * 60)

    # ---- 載入模型 ----
    model, tokenizer = load_sft_model(args.base_model, args.model, args.use_4bit)

    # ---- Prompt 配置 ----
    system_prompt, build_prompt_fn = get_prompt_builder(args.config)

    # ---- 載入測試資料 ----
    print(f"\n📂 載入測試資料: {args.data_path}")
    test_data = load_jsonl(args.data_path)
    total_count = len(test_data)
    print(f"   共 {total_count} 筆測試資料")

    # ---- 初始化 log ----
    log_path = init_log(f"logs/test_sft_{args.config}_log.jsonl")
    print(f"📝 推論記錄: {log_path}")

    # ---- 逐筆推論 ----
    submission_data = []

    for i, sample in enumerate(test_data):
        sample_id = sample["id"]
        user_prompt, valid_options = build_prompt_fn(sample)

        # 本地推論
        try:
            response_text = generate_answer(
                model, tokenizer, system_prompt, user_prompt,
                max_new_tokens=args.max_new_tokens,
            )
            error_msg = None
        except Exception as e:
            response_text = None
            error_msg = str(e)

        # 擷取答案
        if response_text is not None:
            answer, confidence = extract_answer(response_text, valid_options)
            if answer is None:
                answer = heuristic_fallback(sample, valid_options)
                confidence = "heuristic"
        else:
            answer = heuristic_fallback(sample, valid_options)
            confidence = "gen_failed"

        # 記錄
        log_record = {
            "timestamp": now_iso(),
            "id": sample_id,
            "method": "sft",
            "config": args.config,
            "user_prompt": user_prompt,
            "raw_response": response_text,
            "extracted_answer": answer,
            "confidence": confidence,
            "error": error_msg,
        }
        append_log(log_path, log_record)

        submission_data.append({"id": sample_id, "answer": answer})

        # 進度顯示
        if i == 0 or (i + 1) % 100 == 0 or (i + 1) == total_count:
            print(
                f"  [{i+1}/{total_count}] ID={sample_id} → {answer} "
                f"(conf={confidence})",
                flush=True,
            )

    # ---- 輸出 submission.csv ----
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(args.output, index=False)

    print("\n" + "=" * 60)
    print(f"✅ 推論完成！共 {total_count} 筆")
    print(f"📄 Kaggle 上傳檔: {args.output}")
    print(f"💾 推論記錄: {log_path}")
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
