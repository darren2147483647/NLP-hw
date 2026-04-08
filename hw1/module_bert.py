import torch.nn as nn
from transformers import AutoModel

class BertWithCustomHead(nn.Module):
    def __init__(self, model_name, num_labels=12, hidden_dim=128, dropout_rate=0.0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden_size = self.bert.config.hidden_size  # 1024 for BERT-Large
        
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_labels)
        )
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token 經過 Pooler 的輸出
        logits = self.classifier(pooled_output)
        return logits
