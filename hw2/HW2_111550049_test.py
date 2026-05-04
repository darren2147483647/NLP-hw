"""
test5.py
使用訓練好的 Method 5 架構推論 test.json，產出 submission5.csv
嚴格遵守計畫，不擅自加入自訂 DataLoader Collator。
"""

import argparse
import time
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from HW2_111550049_utils import set_seed, save_csv, load_withlora
from HW2_111550049_dataloader import SiameseDataset, SiameseCollate
from HW2_111550049_model import LlamaSiamese

def main():
    parser = argparse.ArgumentParser(description="Siamese Network Inference")
    parser.add_argument("--data", type=str, default="test.json")
    parser.add_argument("--model_id", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--output", type=str, default="submission5.csv")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--prompt", type=str, default="normal", help="System prompt")
    parser.add_argument("--prompt_format", type=str, default="llama", help="Format of prompt engineer")
    args = parser.parse_args()

    if args.prompt == "None":
        args.prompt = None
    if args.prompt_format == "None":
        args.prompt_format = None

    set_seed(args.seed)
    base_dir = Path(__file__).parent
    data_path = str(base_dir / args.data)
    ckpt_path = base_dir / args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found!")

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Creating DataLoader (Strict Static Padding)...")
    test_dataset = SiameseDataset(data_path, tokenizer, max_length=1024, prompt=args.prompt, prompt_format=args.prompt_format)
    
    # 建立動態 Padding 專用的 Collate Function
    collate_fn = SiameseCollate(pad_token_id=tokenizer.pad_token_id)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print("Loading Siamese Base Model...")
    model = LlamaSiamese(model_id=args.model_id, hidden_dim=128, dropout_rate=0.0, quant_bits=4, device=device)
    
    print("Loading Trained Checkpoints (LoRA & Classifier)...")
    load_withlora(model, str(ckpt_path))

    # model.to(device)
    model.eval()

    label_names = {0: "A", 1: "B", 2: "tie", 3: "neither"}
    ids = []
    predictions = []

    print("\nStarting Inference...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids_A = batch["input_ids_A"].to(device)
            attention_mask_A = batch["attention_mask_A"].to(device)
            input_ids_B = batch["input_ids_B"].to(device)
            attention_mask_B = batch["attention_mask_B"].to(device)
            batch_ids = batch["id"]
            
            logits = model(
                input_ids_A=input_ids_A,
                attention_mask_A=attention_mask_A,
                input_ids_B=input_ids_B,
                attention_mask_B=attention_mask_B
            )
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            for id_val, pred_idx in zip(batch_ids, preds):
                ids.append(id_val.item() if hasattr(id_val, "item") else id_val)
                predictions.append(label_names[pred_idx])

    total_time = time.time() - start_time
    print(f"Done! Total Time: {total_time:.2f}s")
    
    out_path = str(base_dir / args.output)
    save_csv(ids, predictions, out_path)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
