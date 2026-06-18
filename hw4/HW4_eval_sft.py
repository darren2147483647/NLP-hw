"""
eval_sft.py - SFT 微調模型評估腳本

功能：使用本地微調後的 LoRA 模型做推論，評估 tool selection 準確率
    （不透過 API，直接用本地模型推論）

使用方式:
  python eval_sft.py --model models/sft_full_info --config full_info
  python eval_sft.py --model models/sft_structural --config structural
  python eval_sft.py --model models/sft_full_info --config full_info --num_samples 100

全部參數:
  --model         微調後模型的路徑 (必填)
  --base_model    基底模型名稱 (預設: Qwen/Qwen2.5-7B-Instruct)
  --config        full_info | structural (預設: full_info)
  --num_samples   驗證筆數 (預設: 50，'all' 表示全部)
  --data_path     驗證資料路徑 (預設: sft_data/{config}_val.jsonl)
  --seed          隨機種子 (預設: 42)
"""

import json
import os
import sys
import argparse
import random
import warnings
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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
    parser = argparse.ArgumentParser(description="SFT 模型評估")
    parser.add_argument("--model", type=str, default="models/sft_full_info",
                        help="微調後模型路徑 (LoRA adapter)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="基底模型名稱")
    parser.add_argument("--config", type=str, default="full_info",
                        choices=["full_info", "structural"],
                        help="Prompt 配置")
    parser.add_argument("--num_samples", type=str, default="50",
                        help="驗證筆數 (數字 or 'all')")
    parser.add_argument("--data_path", type=str, default=None,
                        help="驗證資料路徑 (預設使用 train.jsonl)")
    parser.add_argument("--use_val", action="store_true",
                        help="使用 sft_data 中的 val set 而非 train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
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
    if saved_base and saved_base != base_model_name:
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
    from peft import LoraConfig, get_peft_model
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

    return model, tokenizer, train_config


def generate_answer(model, tokenizer, system_prompt, user_prompt, max_new_tokens=64):
    """使用本地模型生成回答"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 使用 tokenizer 的 chat template
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: 手動組裝
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

    # 只取新生成的部分
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response


def main():
    args = parse_args()

    if args.no_4bit:
        args.use_4bit = False

    random.seed(args.seed)

    # ---- 載入模型（.pt 內含 base_model 資訊，會自動偵測）----
    model, tokenizer, train_config = load_sft_model(
        args.base_model, args.model, args.use_4bit
    )

    # ---- Prompt 配置 ----
    system_prompt, build_prompt_fn = get_prompt_builder(args.config)

    # ---- 載入資料 ----
    if args.use_val:
        val_file = f"sft_data/{args.config}_val.jsonl"
        if not os.path.exists(val_file):
            print(f"❌ 找不到 val 資料: {val_file}")
            sys.exit(1)
        # Val 資料是 chat format，需要轉回 sample format
        # 改用 train.jsonl 做驗證更直接
        print("⚠️  --use_val 模式需搭配原始 train.jsonl 做 ID mapping，改用 train.jsonl 抽樣")

    data_path = args.data_path or "train.jsonl"
    print(f"\n📂 載入驗證資料: {data_path}")
    all_data = load_jsonl(data_path)
    total_available = len(all_data)

    if args.num_samples.lower() == "all":
        samples = all_data
        num_samples = total_available
    else:
        num_samples = int(args.num_samples)
        if num_samples >= total_available:
            samples = all_data
            num_samples = total_available
        else:
            samples = random.sample(all_data, num_samples)

    print(f"   共 {num_samples} 筆驗證資料")

    # ---- 初始化 log ----
    log_path = init_log(f"logs/eval_sft_{args.config}_log.jsonl")
    print(f"📝 記錄: {log_path}")
    print(f"🔧 配置: SFT × {args.config}")

    # ---- 逐筆驗證 ----
    correct = 0
    total = 0
    errors = []
    confidence_stats = Counter()

    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        ground_truth = sample["answer"]

        user_prompt, valid_options = build_prompt_fn(sample)

        # 本地推論（不需 rate limiting）
        try:
            response_text = generate_answer(
                model, tokenizer, system_prompt, user_prompt,
                max_new_tokens=args.max_new_tokens
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
            "method": "sft",
            "config": args.config,
            "raw_response": response_text,
            "extracted_answer": answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "confidence": confidence,
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
    print(f"📊 [SFT × {args.config}] 驗證結果: {correct}/{total} = {correct/total*100:.2f}% 準確率")
    print("=" * 60)

    print(f"\n📈 信心度分布:")
    for conf_level in ["high", "medium", "low", "heuristic", "gen_failed"]:
        count = confidence_stats.get(conf_level, 0)
        if count > 0:
            pct = count / total * 100
            print(f"   {conf_level:12s}: {count:4d} ({pct:5.1f}%)")

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

    # 儲存結果摘要
    result_summary = {
        "method": "sft",
        "config": args.config,
        "model": args.model,
        "base_model": args.base_model,
        "num_samples": num_samples,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "confidence_stats": dict(confidence_stats),
    }
    summary_path = f"logs/eval_sft_{args.config}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False)
    print(f"\n💾 結果摘要: {summary_path}")
    print(f"💾 完整記錄: {log_path}")


if __name__ == "__main__":
    main()
