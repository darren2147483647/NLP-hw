import requests
import base64
import json
import os
import sys
import time
import re
import random
import pandas as pd

QUIET_MODE = True
IMG_DATA_PATH = "images/"

def load_api_key(filepath="api_key.txt"):
    """從 txt 載入 API Key"""
    if not os.path.exists(filepath):
        print(f"找不到 API Key 檔案！請建立 '{filepath}'")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        key = f.read().strip()
        if not key:
            print(f"API Key 檔案 '{filepath}' 是空的！")
            sys.exit(1)
        return key

NVIDIA_API_KEY = load_api_key("api_key.txt")
invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"


def query_gemma_multimodal(content_list):
    """
    發送多模態請求給 Gemma-4-31B-IT。
    content_list: OpenAI Vision 格式的 content 陣列，包含 text 和 image_url 項目。
    """
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json"
    }

    payload = {
        "model": "google/gemma-4-31b-it",
        "messages": [{"role": "user", "content": content_list}],
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.95,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        response = requests.post(invoke_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "")
            if not content:
                return "[Error] 模型回傳了空白內容"
            return content.strip()
        else:
            return "[Error] API 回應格式不符預期 (無 choices)"
    except requests.exceptions.RequestException as e:
        return f"[Error] API 請求失敗: {str(e)}"
    except json.JSONDecodeError:
        return "[Error] 無法解析 API 傳回的 JSON"
    except Exception as e:
        return f"[Error] 發生未知例外: {str(e)}"


def encode_image_to_base64(image_path):
    """讀取本地圖片並轉換為 Base64 字串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_top_5_ids(response_text, valid_quote_ids):
    """從模型回應中擷取最多 5 個合法的 evidence ID，不足時隨機補齊"""
    if not response_text or response_text.startswith("[Error]"):
        pool = list(valid_quote_ids)
        random.shuffle(pool)
        return pool[:5]

    raw_matches = re.findall(r'(text|image)[\s\-_]*(\d+)', response_text.lower())

    valid_ids = []
    seen = set()

    for prefix, num in raw_matches:
        normalized_id = f"{prefix}{int(num)}"

        if normalized_id in valid_quote_ids and normalized_id not in seen:
            valid_ids.append(normalized_id)
            seen.add(normalized_id)
            if len(valid_ids) == 5:
                break

    if len(valid_ids) < 5:
        remaining = list(valid_quote_ids - seen)
        random.shuffle(remaining)
        for rid in remaining:
            valid_ids.append(rid)
            if len(valid_ids) == 5:
                break

    return valid_ids


def build_multimodal_content(question, sample):
    """
    構建多模態 content 陣列 (OpenAI Vision API 格式)。
    - 圖片：以 Base64 編碼的原始圖片傳入，並標註其 ID (如 [image1])
    - 文字：以純文字傳入，並標註其 ID (如 [text1])
    
    回傳: (content_list, valid_quote_ids)
    """
    content_parts = []
    valid_quote_ids = set()

    instruction = (
        "You are an expert retrieval assistant. "
        "I will provide you with a question and a list of evidence items. "
        "Each evidence item has a unique ID (e.g., text1, image3). "
        "Text evidence is provided as plain text. "
        "Image evidence is provided as actual images — each image is labeled with its ID.\n\n"
        "Your task: Analyze ALL evidence items (both text and images) and determine "
        "which 5 are most relevant to answering the question.\n\n"
        f"Question: {question}\n\n"
        "Evidence Items:\n"
    )
    content_parts.append({"type": "text", "text": instruction})

    if 'img_quotes' in sample:
        for iq in sample['img_quotes']:
            quote_id = iq['quote_id']
            img_path = os.path.join(IMG_DATA_PATH, iq['img_path'])
            valid_quote_ids.add(quote_id)

            content_parts.append({"type": "text", "text": f"[{quote_id}]:"})

            if os.path.exists(img_path):
                b64_image = encode_image_to_base64(img_path)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                })
            else:
                # 圖片不存在時用文字描述作為 fallback
                desc = iq.get('img_description', '[Image file not found]')
                content_parts.append({"type": "text", "text": f"(Image not available. Description: {desc})"})

    if 'text_quotes' in sample:
        text_evidence_lines = []
        for tq in sample['text_quotes']:
            quote_id = tq['quote_id']
            valid_quote_ids.add(quote_id)
            text_evidence_lines.append(f"[{quote_id}]: {tq['text']}")

        text_block = "\n".join(text_evidence_lines)
        content_parts.append({"type": "text", "text": text_block})

    final_instruction = (
        "\n\nYou MUST extract and rank EXACTLY 5 evidence IDs in descending order of importance. "
        "Even if you think fewer than 5 items are relevant, you MUST fill all 5 spots with your best guesses. "
        "DO NOT output fewer than 5 IDs. Output ONLY the 5 IDs separated by commas (e.g., text1, image3, text5, image2, text10)."
    )
    content_parts.append({"type": "text", "text": final_instruction})

    return content_parts, valid_quote_ids


def main():
    print("載入 test.jsonl 資料...")
    try:
        with open("test.jsonl", "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
    except Exception as e:
        print(f"讀取 test.jsonl 發生錯誤: {e}")
        return

    results = []
    submission_data = []
    total_count = len(lines)
    last_request_time = 0.0

    for i, sample in enumerate(lines):
        q_id = sample['q_id']
        question = sample['question']

        content_list, valid_quote_ids = build_multimodal_content(question, sample)

        if not QUIET_MODE:
            print(f"\n[{i+1}/{total_count}] Q_ID: {q_id}")
            print(f"問題: {question}")
            print(f"提供知識數量: {len(valid_quote_ids)} 筆")
            print("發送 API 請求中 (包含圖片)...")
        else:
            if i == 0 or (i + 1) % 10 == 0 or (i + 1) == total_count:
                print(f"處理進度: [{i+1}/{total_count}]", end="\r", flush=True)

        # 限制每兩秒只能發送一次請求
        while time.time() - last_request_time < 1.6: # 改成1.6 因為rpm有40
            time.sleep(0.1)

        response_text = query_gemma_multimodal(content_list)
        last_request_time = time.time()

        # 顯示原始回應
        display_text = response_text[:200] + "..." if len(response_text) > 200 else response_text
        if not QUIET_MODE:
            print(f"模型原始回應: {display_text.strip()}")

        # 擷取 Top-5 ID
        predicted_quotes = extract_top_5_ids(response_text, valid_quote_ids)
        if not QUIET_MODE:
            print(f"解析後的 Top-5 排序: {predicted_quotes}")

        # 準備 Kaggle submission 格式
        predicted_str = " ".join(predicted_quotes)
        submission_data.append({
            "q_id": q_id,
            "gold_quotes": predicted_str
        })

        results.append({
            "q_id": q_id,
            "question": question,
            "predicted_quotes": predicted_quotes,
            "model_raw_response": response_text
        })

    # 儲存詳細結果 (Backup)
    output_file = "gemma_answering_img_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 儲存 Kaggle 格式的 Submission
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv("submission.csv", index=False)

    print("\n" + "=" * 50)
    print(f"處理完成！共處理 {total_count} 筆測試資料。")
    print(f"Kaggle 上傳檔已儲存至 submission.csv")
    print(f"詳細回應已備份至 {output_file}")


if __name__ == "__main__":
    main()
