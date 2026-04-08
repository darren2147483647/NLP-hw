import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from module import LSTM
from dataset import get_dataloader
from utils import save, load, set_seed, load_lstm, EMBEDDING_DIM, CLASSES

# 訓練模型
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, lr, device):
    best_acc = 0.0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for i, (texts, labels) in enumerate(tqdm(train_loader, desc=f'Training {epoch+1}/{num_epochs}')):
            optimizer.zero_grad()
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {train_loss/len(train_loader)}')
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for i, (texts, labels) in enumerate(tqdm(val_loader, desc='Validation')):
                    texts, labels = texts.to(device), labels.to(device)
                    outputs = model(texts)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (predicted == labels).sum().item()/len(CLASSES)
                    total += labels.size(0)
                acc = correct/total
                print(f'Accuracy: {acc} ({correct}/{total})')
                if acc > best_acc:
                    best_acc = acc
                    print(f'Epoch {epoch+1}, New best accuracy: {best_acc}')
                    save(model, 'best_model.pth')
    save(model, f'model_{num_epochs}.pth')

if __name__ == '__main__':
    set_seed(69)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyper para
    num_epochs = 200
    batch_size = 32
    lr = 0.001

    # 載入資料
    word_vectors, EMBEDDING_DIM = load_lstm()
    train_loader = get_dataloader("HW1_dataset/train.json", word_vectors=word_vectors, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader("HW1_dataset/val.json", word_vectors=word_vectors, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = LSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=128, output_dim=len(CLASSES))
    model.to(device)

    # 定義損失函數和優化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, lr, device)
