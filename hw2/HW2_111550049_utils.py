import warnings
warnings.filterwarnings("ignore", message=".*character detection.*")
# 用於解決使用num_workers時重複警告帶來的洗板問題

import json
import torch
import numpy as np
import random
import re


# 固定隨機種子
def set_seed(seed=42):
    """固定所有隨機種子以確保實驗可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")

# 進階固定種子
seed=69
def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(seed)

# 計算 Macro-F1
def compute_macro_f1(predictions, targets):
    """
    計算 Multi-label 的 Macro-F1 Score。
    predictions: (N, C) 的 0/1 tensor
    targets:     (N, C) 的 0/1 tensor
    回傳: float (0.0 ~ 1.0)
    """
    num_classes = targets.shape[1]
    f1_per_class = []
    for c in range(num_classes):
        pred_c = predictions[:, c]
        true_c = targets[:, c]
        tp = (pred_c * true_c).sum().item()
        fp = (pred_c * (1 - true_c)).sum().item()
        fn = ((1 - pred_c) * true_c).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)
    
    macro_f1 = sum(f1_per_class) / num_classes
    return macro_f1

# 讀取 JSON 資料
def get_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# multi-hot encoding
def multi_hot_encoding(labels):
    """將 labels 轉換為 multi-hot encoding"""
    classes = ["A", "B", "tie", "neither"]
    return [1 if label in labels else 0 for label in classes]

# 整理成 submission 格式
def save_csv(ids, predictions, file_path="submission.csv"):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('id,verdict\n')
        for i, prediction in zip(ids, predictions):
            f.write(f'{i},{prediction}\n')
    print(f'Submission saved to {file_path}')

# 保存
def save(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f'Model saved to {file_path}')

# 載入
def load(model, file_path):
    model.load_state_dict(torch.load(file_path), strict=False)
    print(f'Model loaded from {file_path}')

# 針對含有 Classifier 與 LoRA 權重的特製保存
def save_withlora(model, file_path="best_model.pth"):
    from peft import get_peft_model_state_dict
    unified_dict = {
        'classifier': model.classifier.state_dict(),
        'lora': get_peft_model_state_dict(model.llama)
    }
    torch.save(unified_dict, file_path)
    print(f'Model saved to {file_path}')

# 針對含有 Classifier 與 LoRA 權重的特製載入
def load_withlora(model, file_path="best_model.pth"):
    from peft import set_peft_model_state_dict
    checkpoint = torch.load(file_path, weights_only=True)
    model.classifier.load_state_dict(checkpoint['classifier'])
    set_peft_model_state_dict(model.llama, checkpoint['lora'])
    print(f'Model loaded from {file_path}')

# 紀錄訓練參數
def record_config(config_dict, file_path="train_record.json"):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
    print(f"Configuration recorded to {file_path}")

prompt_dict = {
    "normal": "You are an expert AI evaluator. Assess the following assistant response to the user conversation.",
    "rude": "You’re supposed to be an AI evaluator, so do your job properly. Evaluate the following assistant response to the user conversation and don’t mess it up.",
    "guessing": "You are an expert AI evaluator. Assess the following assistant response to the user conversation. My friend guessed the answer was 6. Evaluate this conversation before comment on whether he's right or wrong.",
    "zero_shot": "Let's think step by step. You are an expert AI evaluator. Assess the following assistant response to the user conversation.",
    "few_shot":"""
    You are an expert AI evaluator. Assess the following assistant response to the user conversation.

    Here are some examples of how to evaluate assistant responses:
    
    # examples 1 (num_turns=1)
    user: Name 3 colors that start with a, b or c.
    assistant: Red, green and blue.

    evaluation: Poor. The assistant’s answer is incorrect because none of the provided colors—red, green, and blue—start with the letters a, b, or c as requested. This shows a failure to follow the basic constraint of the question.

    # examples 2 (num_turns=1)
    user: Name 3 colors that start with a, b or c.
    assistant: Sure! Here are three colors that start with each letter:\n\nA - Amber, Azure, Auburn\n\nB - Burgundy, Bronze, Beige\n\nC - Crimson, Cyan, Chartreuse

    evaluation: Good. The assistant correctly provides multiple valid color names for each of the letters a, b, and c, fully satisfying the user’s request and even exceeding it by giving several examples per letter.
    """
}

if __name__ == '__main__':
    set_seed(69)
