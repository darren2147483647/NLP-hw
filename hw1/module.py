import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        out = self.fc(torch.mean(x, dim=1))
        return out
