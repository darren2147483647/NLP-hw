import torch
import torch.nn as nn
from transformers import AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

class GELU(nn.Module):
    def forward(self, x):
        return nn.GELU()(x)

class SwiGLU(nn.Module):
    def forward(self, x):
        # 將輸入的特徵切成兩半
        x, gate = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * x

class LlamaSiamese(nn.Module):
    def __init__(self, model_id, num_labels=4, hidden_dim=128, dropout_rate=0.0, quant_bits=4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        self.device = device
        
        # 1. 設置量化設定
        quantization_config = None
        if quant_bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quant_bits == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        print(f"Loading Siamese base model {model_id} in {quant_bits}-bit mode...")
        
        # 2. 載入沒有 lm_head 的預訓練 LLaMA
        self.llama = AutoModel.from_pretrained(
            model_id,
            # device_map="auto",
            device_map={"": self.device},
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if quant_bits in [4, 8] else torch.float32,
        )
        self.llama.config.use_cache = False

        # 3. 準備 LoRA Adapter
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.llama = prepare_model_for_kbit_training(self.llama)
        self.llama = get_peft_model(self.llama, lora_config)
        self.llama.print_trainable_parameters()
        
        # 4. 雙生網路的特徵比較 MLP (Classification Head)
        self.llama_hidden_size = self.llama.config.hidden_size # 4096
        
        # 雙邊特徵 Concat 後維度為 8192
        # 嚴格遵照計畫與您的指示：中間層為 128
        # 架構：Linear(8192, 128 * 2) -> SwiGLU(256切128) -> Dropout -> Linear(128, 4)
        self.classifier = nn.Sequential(
            nn.Linear(self.llama_hidden_size * 2, hidden_dim * 2),
            SwiGLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_labels)
        )
        
        # 確保分類頭是 float32 且允許梯度更新
        self.classifier.to(torch.float32)
        self.classifier.to(self.device)
        for param in self.classifier.parameters():
            param.requires_grad = True

    def _extract_last_token(self, hidden_states, attention_mask, arch_type="decoder"):
        """從 Decoder 的輸出中精確抽取出真實結尾 Token 的 Embeddings"""
        batch_size = hidden_states.shape[0]
        if arch_type == "encoder":
            return hidden_states[:, 0, :] # BERT CLS token
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            sequence_lengths = sequence_lengths.clamp(min=0)
        else:
            sequence_lengths = torch.tensor([hidden_states.shape[1] - 1] * batch_size, device=hidden_states.device)
            
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B):
        """
        雙生網路前向傳播：
        分別運算 A 和 B，再透過 MLP 合併判斷。
        """
        outputs_A = self.llama(input_ids=input_ids_A, attention_mask=attention_mask_A)
        last_hidden_A = self._extract_last_token(outputs_A.last_hidden_state, attention_mask_A)
        
        outputs_B = self.llama(input_ids=input_ids_B, attention_mask=attention_mask_B)
        last_hidden_B = self._extract_last_token(outputs_B.last_hidden_state, attention_mask_B)
        
        # 特徵拼接 (Batch Size, 8192)
        pooled = torch.cat([last_hidden_A, last_hidden_B], dim=-1)
        
        # 分析並輸出 4 種 Logits
        logits = self.classifier(pooled.to(torch.float32))
        return logits

if __name__ == "__main__":
    model = LlamaSiamese("Skywork/Skywork-Reward-V2-Llama-3.1-8B")
    print(model)