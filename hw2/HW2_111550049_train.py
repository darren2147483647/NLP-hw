"""
train5.py
使用 Method 5 雙生網路架構 (Siamese LLaMA + SwiGLU MLP) 重新訓練。
嚴格遵守計畫，不擅自加入自訂 DataLoader Collator。
"""

import argparse
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt

from HW2_111550049_utils import set_seed, save_withlora, record_config, seed_worker, g
from HW2_111550049_dataloader import SiameseDataset, SiameseCollate
from HW2_111550049_model import LlamaSiamese

def main():
    parser = argparse.ArgumentParser(description="Train Siamese LLM Judge directly")
    parser.add_argument("--data", type=str, default="train.json", help="Path to training data")
    parser.add_argument("--model_id", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B")
    parser.add_argument("--output_dir", type=str, default=".", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4, help="Siamese network 載入全 Padding 序列，記憶體消耗極大，建議設1")
    parser.add_argument("--acc_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--prompt", type=str, default="normal", help="System prompt")
    parser.add_argument("--prompt_format", type=str, default="llama", help="Format of prompt engineer")
    parser.add_argument("--ABswitch", type=float, default=0.0, help="Posibility of swapping the position of dialog_1 and dialog_2")
    parser.add_argument("--weightCE", action="store_true", default=False, help="Use weighted CE loss")
    parser.add_argument("--grad_clip_and_norm", type=float, default=1.0, help="Gradient clipping and norm")
    args = parser.parse_args()

    if args.prompt == "None":
        args.prompt = None
    if args.prompt_format == "None":
        args.prompt_format = None
    if args.grad_clip_and_norm == "None" or args.grad_clip_and_norm == "0.0":
        args.grad_clip_and_norm = None

    set_seed(args.seed)
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    os.makedirs(output_dir, exist_ok=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Creating Datasets and Dataloaders (Strict Static Padding)...")
    data_path = str(base_dir / args.data)
    
    train_dataset = SiameseDataset(data_path, tokenizer, max_length=1024, split='train', ABswitch=args.ABswitch, prompt=args.prompt, prompt_format=args.prompt_format)
    val_dataset = SiameseDataset(data_path, tokenizer, max_length=1024, split='val', prompt=args.prompt, prompt_format=args.prompt_format)
    
    # 建立動態 Padding 專用的 Collate Function
    collate_fn = SiameseCollate(pad_token_id=tokenizer.pad_token_id)

    # 嚴格依照原始 Pytorch DataLoader 邏輯運作，加入 collate_fn
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, worker_init_fn=seed_worker, generator=g, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, worker_init_fn=seed_worker, generator=g, collate_fn=collate_fn)

    print("Loading Siamese Model (4-bit QLoRA)...")
    # 按照指示，Hidden Dim 為 128
    model = LlamaSiamese(model_id=args.model_id, hidden_dim=128, dropout_rate=0.0, quant_bits=4, device=device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # 針對標籤不平衡 (A=0, B=1, tie=2, neither=3)，設定對應的損失權重
    class_weights = torch.tensor([1.0, 1.0, 3.5, 1.75], dtype=torch.float32).to(device)
    if args.weightCE:
        smoothed_weights = torch.log(torch.clamp(class_weights, min=1.0)) + 1.0
        criterion = nn.CrossEntropyLoss(weight=smoothed_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    num_training_steps = ((len(train_loader) + args.acc_steps - 1) // args.acc_steps) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio), num_training_steps=num_training_steps)
    # model.to(device)

    start_time = time.time()

    # 紀錄各項參數
    config_dict = {
        "seed": args.seed,
        "MODEL_ID": args.model_id,
        "QUANT_BITS": 4,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulation_steps": args.acc_steps,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "prompt": args.prompt,
        "prompt_format": args.prompt_format,
        "ABswitch": args.ABswitch,
        "weightCE": args.weightCE,
        "grad_clip_and_norm": args.grad_clip_and_norm,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "state": "training"
    }
    record_config(config_dict)
    
    print("\nStarting Training...")
    best_acc = 0.0
    loss_history = []
    acc_history = []
    val_epochs = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            input_ids_A = batch["input_ids_A"].to(device)
            attention_mask_A = batch["attention_mask_A"].to(device)
            input_ids_B = batch["input_ids_B"].to(device)
            attention_mask_B = batch["attention_mask_B"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(
                input_ids_A=input_ids_A,
                attention_mask_A=attention_mask_A,
                input_ids_B=input_ids_B,
                attention_mask_B=attention_mask_B
            )
            
            loss = criterion(logits, labels)
            loss = loss / args.acc_steps
            loss.backward()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * args.acc_steps
            
            if (step + 1) % args.acc_steps == 0 or (step + 1) == len(train_loader):
                if args.grad_clip_and_norm is not None:
                    if args.grad_clip_and_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_and_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.set_postfix({
                "loss": f"{loss.item() * args.acc_steps:.4f}",
                "acc": f"{correct / total:.4f}"
            })
            
        epoch_acc = correct / total
        epoch_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} finished! Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        
        loss_history.append(epoch_loss)
        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids_A = batch["input_ids_A"].to(device)
                attention_mask_A = batch["attention_mask_A"].to(device)
                input_ids_B = batch["input_ids_B"].to(device)
                attention_mask_B = batch["attention_mask_B"].to(device)
                labels = batch["labels"].to(device)
                
                logits = model(
                    input_ids_A=input_ids_A,
                    attention_mask_A=attention_mask_A,
                    input_ids_B=input_ids_B,
                    attention_mask_B=attention_mask_B
                )
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        acc_history.append(val_acc)
        val_epochs.append(epoch + 1)
        
        print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = output_dir / "best_model.pth"
            print(f"New best Accuracy: {best_acc:.4f} ! Saving weights to {best_model_path}...\n")
            save_withlora(model, str(best_model_path))
            
    final_model_path = output_dir / "final_model.pth"
    print(f"Training Complete! Saving final weights to {final_model_path}...")
    save_withlora(model, str(final_model_path))
    
    total_time_mins = (time.time() - start_time) / 60
    print(f"Total Time: {total_time_mins:.2f} mins")

    # 更新訓練狀態
    import json
    try:
        with open("train_record.json", "r", encoding="utf-8") as f:
            record_data = json.load(f)
        if record_data.get("state") == "training":
            record_data["state"] = "done"
            record_data["total_time"] = f"{total_time_mins:.2f} mins"
            record_data["val_score"] = best_acc
            with open("train_record.json", "w", encoding="utf-8") as f:
                json.dump(record_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to update train_record.json: {e}")

        # ====== 產生訓練曲線圖 ======
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, args.epochs + 1), loss_history, color='tab:red', alpha=0.7, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(val_epochs, acc_history, color='tab:blue', marker='o', label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title('Training Loss & Validation Accuracy (LLaMA QLoRA)')
    fig.tight_layout()
    plt.savefig('training_curve_llama.png', dpi=150)
    print('訓練曲線已儲存至 training_curve_llama.png')

if __name__ == "__main__":
    main()
