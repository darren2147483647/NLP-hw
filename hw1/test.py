import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from module import LSTM
from dataset import get_dataloader
from utils import save, load, set_seed, load_lstm, EMBEDDING_DIM, CLASSES, save_csv

# 訓練模型
def test(model, test_loader, device):
    ids_ = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for i, (texts, ids) in enumerate(tqdm(test_loader, desc='Testing')):
            texts = texts.to(device)
            outputs = model(texts)
            predicted = (torch.sigmoid(outputs) > 0.5).int()
            for ii in range(len(ids)):
                predictions.append(predicted[ii].tolist())
                ids_.append(ids[ii])
    save_csv(ids_, predictions)
            
if __name__ == '__main__':
    set_seed(69)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyper para
    num_epochs = 20
    batch_size = 32

    # 載入資料
    word_vectors, EMBEDDING_DIM = load_lstm()
    test_loader = get_dataloader("HW1_dataset/test.json", word_vectors=word_vectors, batch_size=batch_size, shuffle=False, test=True)
    # 載入模型
    model = LSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=128, output_dim=len(CLASSES))
    load(model, 'best_model.pth')
    model.to(device)

    test(model, test_loader, device)
