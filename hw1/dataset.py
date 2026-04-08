import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import process_json, load_lstm

class TweetDataset(Dataset):
    def __init__(self, file_path, word_vectors=None, test=False):
        self.ids, self.texts, self.labels = process_json(file_path, word_vectors, no_label=test)
        self.test = test

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if self.test:
            return torch.tensor(self.texts[idx], dtype=torch.float32), self.ids[idx]
        return torch.tensor(self.texts[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# 取得dataloader
def get_dataloader(file_path, word_vectors, batch_size=32, shuffle=True, test=False):
    dataset = TweetDataset(file_path, word_vectors, test=test)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=shuffle)

if __name__ == '__main__':
    word_vectors, EMBEDDING_DIM = load_lstm()
    train_loader = get_dataloader("HW1_dataset/train.json", word_vectors=word_vectors)
    val_loader = get_dataloader("HW1_dataset/val.json", word_vectors=word_vectors)
