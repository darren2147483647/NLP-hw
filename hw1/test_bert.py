import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset_bert import get_bert_dataloader
from utils import set_seed, save_csv, CLASSES

from module_bert import BertWithCustomHead

DATA_DIR = "."

def test_bert(model, test_loader, device):
    ids_ = []
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, batch_ids) in enumerate(tqdm(test_loader, desc='Testing')):
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs
            
            predicted = (torch.sigmoid(logits) > 0.5).int()
            
            for ii in range(len(batch_ids)):
                predictions.append(predicted[ii].tolist())
                ids_.append(batch_ids[ii])
                
    save_csv(ids_, predictions, file_path="submission_bert.csv")
            
if __name__ == '__main__':
    set_seed(69)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    MODEL_NAME = "digitalepidemiologylab/covid-twitter-bert-v2"
    
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    MODEL_PATH = "best_bert_model.pth"#"best_bert_model.pth"
    
    model = BertWithCustomHead(MODEL_NAME, num_labels=len(CLASSES), hidden_dim=128)
    print(f"從 {MODEL_PATH} 載入模型...")
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)
    model.to(device)

    print("載入測試資料集...")
    test_loader = get_bert_dataloader(os.path.join(DATA_DIR, "HW1_dataset", "test.json"), tokenizer, batch_size=batch_size, shuffle=False, test=True)

    print("開始測試推論...")
    test_bert(model, test_loader, device)
