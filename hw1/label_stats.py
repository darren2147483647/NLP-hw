import json

CLASSES = ['ineffective','unnecessary','pharma','rushed','side-effect','mandatory','country','ingredients','political','none','conspiracy','religious']

with open('HW1_dataset/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

total = len(data)
pos_count = {c: 0 for c in CLASSES}

for item in data:
    for label in item['labels'].keys():
        if label in pos_count:
            pos_count[label] += 1

print(f"總樣本數: {total}\n")
print(f"{'類別':<15} {'正類':>6} {'負類':>6} {'正類比例':>8} {'負/正比':>8}  建議 pos_weight")
print("-" * 75)

pos_weights = []
for c in CLASSES:
    pos = pos_count[c]
    neg = total - pos
    ratio = pos / total
    neg_pos_ratio = neg / pos if pos > 0 else float('inf')
    pos_weights.append(round(neg_pos_ratio, 2))
    print(f"{c:<15} {pos:>6} {neg:>6} {ratio:>8.2%} {neg_pos_ratio:>8.2f}  {neg_pos_ratio:.2f}")

print(f"\n# 可直接複製到 train_bert.py 使用:")
print(f"pos_weight = torch.tensor({pos_weights}).to(device)")
