import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt

from dataset_bert import get_bert_dataloader
from utils import set_seed, CLASSES, compute_macro_f1

from module_bert import BertWithCustomHead

DATA_DIR = "."

ENABLE_WEIGHT_DECAY = True

def train_bert(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, tokenizer):
    best_f1 = 0.0
    loss_history = []
    f1_history = []
    f1_epochs = []
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Training {epoch+1}/{num_epochs}')):
            optimizer.zero_grad()
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs
            
            loss = criterion(logits, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 讓學習率根據 warmup 排程更新
            
        avg_loss = train_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')
        
        # 每隔幾個 Epoch 跑一次驗證
        if (epoch + 1) % 2 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(tqdm(val_loader, desc='Validation')):
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    labels = labels.to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs
                    
                    predicted = (torch.sigmoid(logits) > 0.5).float()
                    all_preds.append(predicted)
                    all_labels.append(labels)
                    
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            macro_f1 = compute_macro_f1(all_preds, all_labels)
            f1_history.append(macro_f1)
            f1_epochs.append(epoch + 1)
            
            print(f'Macro-F1: {macro_f1:.4f}')
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                print(f'Epoch {epoch+1}, New best Macro-F1: {best_f1:.4f} ! Saving...')
                
                torch.save(model.state_dict(), 'best_bert_model.pth')
    torch.save(model.state_dict(), f'bert_model_{num_epochs}')
    
    # ====== 產生訓練曲線圖 ======
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左軸：Loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, num_epochs + 1), loss_history, color='tab:red', alpha=0.7, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    # 右軸：Macro-F1
    ax2 = ax1.twinx()
    ax2.set_ylabel('Macro-F1', color='tab:blue')
    ax2.plot(f1_epochs, f1_history, color='tab:blue', marker='o', label='Val Macro-F1')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title('Training Loss & Validation Macro-F1')
    fig.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    print('訓練曲線已儲存至 training_curve.png')
                
if __name__ == '__main__':
    # 固定種子確保重現性
    set_seed(69)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 8
    batch_size = 16
    lr = 5e-5
    warmup_ratio = 0.1
    
    MODEL_NAME = "digitalepidemiologylab/covid-twitter-bert-v2"
    
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Preparing Dataloaders...")
    train_loader = get_bert_dataloader(os.path.join(DATA_DIR, "HW1_dataset", "train.json"), tokenizer, batch_size=batch_size, shuffle=True)
    val_loader = get_bert_dataloader(os.path.join(DATA_DIR, "HW1_dataset", "val.json"), tokenizer, batch_size=batch_size, shuffle=False)

    print(f"Loading model: {MODEL_NAME}...")
    model = BertWithCustomHead(MODEL_NAME, num_labels=len(CLASSES), hidden_dim=128)
    model.to(device)

    # ====== 分層學習率 (Discriminative LR) ======
    # 核心概念：越底層的 BERT 層學到越通用的特徵，不需要大幅更新 → 給小 LR
    #           越頂層的 BERT 層學到越任務相關的特徵 → 給大 LR
    #           classifier 分類頭是從零學起 → 給最大的 LR
    # 只需要調 lr_decay 一個參數！(通常 0.9~0.95 之間效果好)
    lr_decay = 0.95  # 每往下一層，LR 乘以 0.95
    lr_bert_ratio = 0.9

    param = None
    if ENABLE_WEIGHT_DECAY:
        param_groups = []
        num_layers = 24  # BERT-Large 有 24 層
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        
        for layer_idx in range(num_layers):
            layer_lr = lr * lr_bert_ratio * (lr_decay ** (num_layers - layer_idx))
            param_groups.append({
                'params': model.bert.encoder.layer[layer_idx].parameters(),
                'lr': layer_lr
            })

        param_groups.append({
            'params': model.bert.pooler.parameters(),
            'lr': lr * lr_bert_ratio
        })
        
        param_groups.append({
            'params': model.classifier.parameters(),
            'lr': lr
        })
        
        # 印出各層 LR 概覽
        print(f"\n【分層學習率設定】lr_decay = {lr_decay}")
        print(f"  Embeddings:    {0.0:.2e}")
        print(f"  Layer 0 (底):  {lr * lr_bert_ratio * (lr_decay ** num_layers):.2e}")
        print(f"  Layer 23 (頂): {lr * lr_bert_ratio * (lr_decay ** 1):.2e}")
        print(f"  Classifier:    {lr:.2e}")

        param = param_groups
    else:
        param = model.parameters()

    pos_weight = torch.tensor([4.94, 12.83, 6.82, 5.75, 1.61, 11.69, 48.69, 21.88, 14.92, 14.81, 19.4, 153.58]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.clamp(torch.log1p(pos_weight), min=1.0))
    optimizer = optim.AdamW(param, lr=lr)
    
    # 計算總步數與 warmup 步數
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"【Scheduler設定】總步數: {total_steps}, Warmup步數: {warmup_steps}")
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    train_bert(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, tokenizer)
