"""
train_sft.py - SFT LoRA 微調腳本

功能：使用 QLoRA 在 train.jsonl 上微調 LLM 做 tool selection

使用方式:
  python train_sft.py --config full_info                          # Full-Information 訓練
  python train_sft.py --config structural                         # Structural-Only 訓練
  python train_sft.py --config full_info --model Qwen/Qwen2.5-3B-Instruct  # 指定模型

全部參數:
  --model         模型名稱/路徑 (預設: Qwen/Qwen2.5-7B-Instruct)
  --config        full_info | structural (預設: full_info)
  --data_dir      SFT 資料目錄 (預設: sft_data)
  --output_dir    模型輸出目錄 (預設: models/sft_{config})
  --epochs        訓練 epoch 數 (預設: 3)
  --batch_size    每 GPU batch size (預設: 4)
  --grad_accum    梯度累積步數 (預設: 4)
  --lr            學習率 (預設: 2e-4)
  --lora_rank     LoRA rank (預設: 16)
  --lora_alpha    LoRA alpha (預設: 32)
  --max_seq_len   最大序列長度 (預設: 1024)
  --seed          隨機種子 (預設: 42)
"""

import os
import sys
import json
import argparse
import warnings
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 抑制洗版 warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT LoRA 微調")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="基底模型名稱 (預設: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--config", type=str, default="full_info",
                        choices=["full_info", "structural"],
                        help="Prompt 配置 (預設: full_info)")
    parser.add_argument("--data_dir", type=str, default="sft_data",
                        help="SFT 資料目錄")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="模型輸出目錄 (預設: models/sft_{config})")
    parser.add_argument("--epochs", type=int, default=3,
                        help="訓練 epoch 數")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="每 GPU batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="梯度累積步數")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="學習率")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="最大序列長度")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機種子")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="使用 4-bit 量化 (QLoRA)")
    parser.add_argument("--no_4bit", action="store_true",
                        help="不使用 4-bit 量化")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_4bit:
        args.use_4bit = False

    if args.output_dir is None:
        args.output_dir = f"models/sft_{args.config}"

    print("=" * 60)
    print("🚀 SFT LoRA 微調")
    print("=" * 60)
    print(f"   模型: {args.model}")
    print(f"   配置: {args.config}")
    print(f"   資料: {args.data_dir}")
    print(f"   輸出: {args.output_dir}")
    print(f"   QLoRA: {'Yes (4-bit)' if args.use_4bit else 'No'}")
    print(f"   LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"   Epochs={args.epochs}, BS={args.batch_size}, GA={args.grad_accum}")
    print(f"   LR={args.lr}, MaxLen={args.max_seq_len}")
    print("=" * 60)

    # ---- 確定資料路徑 ----
    if args.config == "full_info":
        train_file = os.path.join(args.data_dir, "full_info_train.jsonl")
        val_file = os.path.join(args.data_dir, "full_info_val.jsonl")
    else:
        train_file = os.path.join(args.data_dir, "structural_train.jsonl")
        val_file = os.path.join(args.data_dir, "structural_val.jsonl")

    if not os.path.exists(train_file):
        print(f"❌ 找不到訓練資料: {train_file}")
        print("   請先執行 python prepare_sft_data.py")
        sys.exit(1)

    # ---- 載入資料 ----
    print("\n📂 載入訓練資料...")
    dataset = load_dataset("json", data_files={
        "train": train_file,
        "validation": val_file,
    })
    print(f"   Train: {len(dataset['train'])} 筆")
    print(f"   Val:   {len(dataset['validation'])} 筆")

    # ---- 載入 Tokenizer ----
    print(f"\n📦 載入 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- 量化配置 ----
    if args.use_4bit:
        print("🔧 配置 4-bit 量化 (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # ---- 載入模型 ----
    print(f"📦 載入模型: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16 if not args.use_4bit else None,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # ---- LoRA 配置 ----
    print("🔧 配置 LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # ---- 訓練配置 ----
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        seed=args.seed,
        max_length=args.max_seq_len,
        report_to="none",
    )

    # ---- 訓練 ----
    print("\n🏋️ 開始訓練...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()

    # ---- 儲存 ----
    print(f"\n💾 儲存模型到: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. HuggingFace 格式 (adapter_model.safetensors + adapter_config.json) 作為備份
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 2. 匯出 .pt 格式 — 單一檔案包含所有需要的資訊
    lora_state_dict = {
        k: v.cpu() for k, v in trainer.model.named_parameters() if v.requires_grad
    }

    train_config = {
        "base_model": args.model,
        "config": args.config,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "max_seq_len": args.max_seq_len,
        "use_4bit": args.use_4bit,
        "train_size": len(dataset["train"]),
        "val_size": len(dataset["validation"]),
    }

    lora_config_dict = {
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }

    pt_path = os.path.join(args.output_dir, "lora_weights.pt")
    torch.save({
        "lora_state_dict": lora_state_dict,
        "lora_config": lora_config_dict,
        "train_config": train_config,
    }, pt_path)

    # 也單獨儲存 train_config.json（方便人工查看）
    config_json_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_json_path, "w") as f:
        json.dump(train_config, f, indent=2)

    print(f"\n   📦 HuggingFace 格式: {args.output_dir}/adapter_model.safetensors")
    print(f"   📦 .pt 權重檔: {pt_path}")
    print(f"   📋 訓練配置: {config_json_path}")

    print("\n" + "=" * 60)
    print("✅ 訓練完成！")
    print(f"   模型路徑: {args.output_dir}")
    print(f"   權重檔案: {pt_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
