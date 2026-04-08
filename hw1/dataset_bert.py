import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np

from utils import get_json, multi_hot_encoding, CLASSES

class BertTweetDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128, test=False):
        self.test = test
        data = get_json(file_path)
        
        self.ids = []
        self.texts = []
        self.labels = []
        
        for item in data:
            self.ids.append(item['ID'])
            self.texts.append(item['tweet'])
            if self.test:
                self.labels.append(np.zeros(len(CLASSES)))
            else:
                self.labels.append(multi_hot_encoding(item['labels'].keys(), CLASSES))
                
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt' # 讓它直接回傳 PyTorch Tensors
        )
        
        # tokenizer 回傳的是 2D tensor (1, max_len)，攤平成 1D 以配合 batching
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        token_type_ids = inputs['token_type_ids'].flatten()
        
        if self.test:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }, self.ids[idx]
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }, label

def get_bert_dataloader(file_path, tokenizer, batch_size=16, shuffle=True, test=False, max_len=128):
    from utils import seed_worker, g
    dataset = BertTweetDataset(file_path, tokenizer, max_len=max_len, test=test)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_loader = get_bert_dataloader("HW1_dataset/train.json", tokenizer)
    val_loader = get_bert_dataloader("HW1_dataset/val.json", tokenizer)
