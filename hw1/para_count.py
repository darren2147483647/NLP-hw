from transformers import AutoModelForSequenceClassification

MODEL_NAME = "digitalepidemiologylab/covid-twitter-bert-v2"
NUM_LABELS = 12

print(f"載入模型: {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

# 總參數量
total_params = sum(p.numel() for p in model.parameters())
# 可訓練參數量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n{'='*50}")
print(f"模型: {MODEL_NAME}")
print(f"{'='*50}")
print(f"總參數量:     {total_params:>15,}")
print(f"可訓練參數量: {trainable_params:>15,}")
print(f"模型大小:     {total_params * 4 / (1024**2):>12.2f} MB (FP32)")
print(f"{'='*50}")

# 各層參數量明細
print(f"\n{'層名稱':<45} {'參數量':>12}")
print("-" * 60)
for name, param in model.named_parameters():
    print(f"{name:<45} {param.numel():>12,}")
