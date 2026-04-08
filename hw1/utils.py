import warnings
warnings.filterwarnings("ignore", message=".*character detection.*")
# 用於解決使用num_workers時重複警告帶來的洗板問題

import json
import torch
import numpy as np
import random
import re

import gensim.downloader as api
word_vectors = None
EMBEDDING_DIM = None
CLASSES = ['ineffective','unnecessary','pharma','rushed','side-effect','mandatory','country','ingredients','political','none','conspiracy','religious']
MAX_LEN = 128

def load_lstm():
    global word_vectors, EMBEDDING_DIM
    word_vectors = api.load("glove-twitter-100")
    EMBEDDING_DIM = word_vectors.vector_size
    return word_vectors, EMBEDDING_DIM

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
def multi_hot_encoding(labels, classes):
    """將 labels 轉換為 multi-hot encoding"""
    return [1 if label in labels else 0 for label in classes]

# 切詞
def tokenize(text, word_vectors, max_len=None):
    text = text.lower()
    text = re.sub(r'@\w+', '@some_user', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    vectors = [word_vectors.get_vector(word, norm=True) if word in word_vectors else np.zeros(word_vectors.vector_size) for word in words]
    if max_len:
        vectors = vectors[:max_len]
        padding = np.zeros((max_len - len(vectors), word_vectors.vector_size))
        vectors = np.vstack((vectors, padding))
    return np.array(vectors)

# 整理資料
def process_json(file_path, word_vectors, classes=CLASSES, max_len=MAX_LEN, no_label=False):
    data = get_json(file_path)
    ids = []
    texts = []
    labels = []
    for item in data:
        ids.append(item['ID'])
        texts.append(tokenize(item['tweet'], word_vectors, max_len=max_len))
        if no_label:
            labels.append(np.zeros(len(classes)))
        else:
            labels.append(multi_hot_encoding(item['labels'].keys(), classes))
    return ids, texts, labels

# 整理成 submission 格式
def save_csv(ids, predictions, file_path="submission.csv"):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('index,ineffective,unnecessary,pharma,rushed,side-effect,mandatory,country,ingredients,political,none,conspiracy,religious\n')
        for i, prediction in zip(ids, predictions):
            f.write(f'{i},{prediction[0]},{prediction[1]},{prediction[2]},{prediction[3]},{prediction[4]},{prediction[5]},{prediction[6]},{prediction[7]},{prediction[8]},{prediction[9]},{prediction[10]},{prediction[11]}\n')
    print(f'Submission saved to {file_path}')

# 保存
def save(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f'Model saved to {file_path}')

# 載入
def load(model, file_path):
    model.load_state_dict(torch.load(file_path), strict=False)
    print(f'Model loaded from {file_path}')

if __name__ == '__main__':
    set_seed(69)
