import torch
from torch.utils.data import Dataset
import json
import random

class SiameseDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024, split=None, ABswitch=None, prompt=None, prompt_format=None):
        """
        供 Siamese Network 使用的 Dataset。
        嚴格遵守計畫：在此處直接把輸入資料轉換成獨立的 input_ids_A 與 input_ids_B。
        """
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if split in ['train', 'val']:
            # 使用獨立的 Random 實例固定 Seed，確保 Train/Val 切分完全一致不重疊
            rng = random.Random(69)
            indices = list(range(len(data)))
            rng.shuffle(indices)
            
            split_idx = int(len(data) * 0.9)
            if split == 'train':
                self.data = [data[i] for i in indices[:split_idx]]
            elif split == 'val':
                self.data = [data[i] for i in indices[split_idx:]]
        else:
            self.data = data

        self.ABswitch = ABswitch
        from llm_utils import prompt_dict
        assert prompt is None or prompt in prompt_dict, f"Prompt '{prompt}' not found."
        self.prompt = prompt_dict[prompt] if prompt else None
        assert prompt_format in ['general', 'json', 'llama', 'double_general'], f"Prompt format '{prompt_format}' not found."
        self.prompt_format = prompt_format
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"A": 0, "B": 1, "tie": 2, "neither": 3}

    def _format_dialog(self, dialog, system_prompt=None):
        if system_prompt:
            system_prompt=f"{system_prompt}\n\n"
        else:
            system_prompt=""
        lines = []
        for msg in dialog:
            role = msg['role'].capitalize()
            lines.append(f"[{role}]: {msg['content']}")
        return system_prompt + "\n".join(lines)

    def _format_dialog_json(self, dialog, system_prompt=None): # 直接把json轉成文字
        if system_prompt:
            system_prompt=f"{system_prompt}\n\n"
        else:
            system_prompt=""
        return system_prompt + json.dumps(dialog)

    def _format_dialog_llama(self, dialog, system_prompt=None):
        if system_prompt is not None:
            dialog.insert(0, {"role": "system", "content": system_prompt})
        # 這會自動套用 LLaMA 3 最喜歡的專屬格式
        return self.tokenizer.apply_chat_template(dialog, tokenize=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 複製對話清單，避免修改到原始資料
        diag_a = item["dialog_1"].copy()
        diag_b = item["dialog_2"].copy()
        
        # 隨機互換對話位置 (True/False機率各50%)
        if self.ABswitch is not None:
            absw = False
            if random.random() < self.ABswitch:
                diag_a, diag_b = diag_b, diag_a
                absw = True

        system_prompt = None
        if self.prompt:
            system_prompt = str(self.prompt)

        if self.prompt_format == 'llama':
            diag_a = self._format_dialog_llama(diag_a, system_prompt)
            diag_b = self._format_dialog_llama(diag_b, system_prompt)
        elif self.prompt_format == 'general':
            diag_a = self._format_dialog(diag_a, system_prompt)
            diag_b = self._format_dialog(diag_b, system_prompt)
        elif self.prompt_format == 'json':
            diag_a = self._format_dialog_json(diag_a, system_prompt)
            diag_b = self._format_dialog_json(diag_b, system_prompt)
        elif self.prompt_format == 'double_general':
            diag_a = diag_a + diag_a
            diag_b = diag_b + diag_b
            diag_a = self._format_dialog(diag_a, system_prompt)
            diag_b = self._format_dialog(diag_b, system_prompt)

        if self.prompt_format == 'llama':
            # 依照原始計畫，進行靜態 Padding 到 max_length
            enc_a = self.tokenizer(
                diag_a, 
                truncation=True, 
                max_length=self.max_length,
                # padding="max_length",
                return_tensors="pt",
                add_special_tokens=False # _format_dialog_llama已經加了，避免在開頭多加一個 <|begin_of_text|>
            )
            enc_b = self.tokenizer(
                diag_b, 
                truncation=True, 
                max_length=self.max_length,
                # padding="max_length",
                return_tensors="pt",
                add_special_tokens=False # _format_dialog_llama已經加了，避免在開頭多加一個 <|begin_of_text|>
            )
        else:
            # 依照原始計畫，進行靜態 Padding 到 max_length
            enc_a = self.tokenizer(
                diag_a, 
                truncation=True, 
                max_length=self.max_length,
                # padding="max_length",
                return_tensors="pt"
            )
            enc_b = self.tokenizer(
                diag_b, 
                truncation=True, 
                max_length=self.max_length,
                # padding="max_length",
                return_tensors="pt"
            )
        
        # 處理測試集缺少的 label
        label_str = item.get("verdict", "tie")
        if self.ABswitch is not None:
            if absw:
                if label_str == "A":
                    label_str = "B"
                elif label_str == "B":
                    label_str = "A"
        label = self.label_map[label_str]

        # 將 Batch 維度消除 (因為 DataLoader 會再次把他們堆疊)
        return {
            "input_ids_A": enc_a["input_ids"].squeeze(0),
            "attention_mask_A": enc_a["attention_mask"].squeeze(0),
            "input_ids_B": enc_b["input_ids"].squeeze(0),
            "attention_mask_B": enc_b["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "id": item.get("id", idx)
        }

from torch.nn.utils.rnn import pad_sequence

class SiameseCollate:
    """
    動態 Padding 的核心：將同一個 Batch 內的長度不一的對話，
    自動補齊至「該 Batch 中最長的長度」，大幅減少無意義的矩陣運算！
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids_A = [item["input_ids_A"] for item in batch]
        attention_mask_A = [item["attention_mask_A"] for item in batch]
        
        input_ids_B = [item["input_ids_B"] for item in batch]
        attention_mask_B = [item["attention_mask_B"] for item in batch]
        
        labels = [item["labels"] for item in batch]
        ids = [item["id"] for item in batch]
        
        # 以 padding_value 補齊序列，batch_first=True 表示輸出形狀為 (Batch, Seq_len)
        padded_input_ids_A = pad_sequence(input_ids_A, batch_first=True, padding_value=self.pad_token_id)
        padded_attention_mask_A = pad_sequence(attention_mask_A, batch_first=True, padding_value=0)
        
        padded_input_ids_B = pad_sequence(input_ids_B, batch_first=True, padding_value=self.pad_token_id)
        padded_attention_mask_B = pad_sequence(attention_mask_B, batch_first=True, padding_value=0)
        
        return {
            "input_ids_A": padded_input_ids_A,
            "attention_mask_A": padded_attention_mask_A,
            "input_ids_B": padded_input_ids_B,
            "attention_mask_B": padded_attention_mask_B,
            "labels": torch.stack(labels),
            "id": ids
        }

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from llm_utils import set_seed #

    # 1. 基本設定
    # 請確保此處的 model_id 與你訓練時一致，才能正確載入 Chat Template
    model_id = "Skywork/Skywork-Reward-V2-Llama-3.1-8B" 
    data_path = "train.json" # 請確保目錄下有這個檔案
    set_seed(69) #

    print("--- 正在初始化 Tokenizer 與 Dataset ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 這裡的設定應與你修正後的邏輯一致
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 實例化 Dataset
    # 測試時先帶入一個存在的 prompt key，例如 "normal"
    try:
        dataset = SiameseDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=1024,
            prompt="normal",
            prompt_format="llama"
        ) #

        # 3. 取得第一筆資料
        sample = dataset[0] #
        
        print("\n" + "="*30)
        print("🔍 驗證結果：Dialog A (還原文字)")
        print("="*30)
        # 使用 decode 將 input_ids 轉回文字，並檢查是否有重複的 BOS
        decoded_a = tokenizer.decode(sample["input_ids_A"], skip_special_tokens=False)
        print(decoded_a)

        print("\n" + "="*30)
        print("🔍 驗證結果：Dialog B (還原文字)")
        print("="*30)
        decoded_b = tokenizer.decode(sample["input_ids_B"], skip_special_tokens=False)
        print(decoded_b)

        print("\n" + "="*30)
        print("📊 數值檢查")
        print("="*30)
        print(f"Label Index: {sample['labels'].item()}")
        print(f"Sequence A Length: {len(sample['input_ids_A'])}")
        print(f"Sequence B Length: {len(sample['input_ids_B'])}")
        
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到資料檔案 {data_path}，請檢查路徑。")
    except Exception as e:
        print(f"❌ 發生錯誤：{e}")