"""
utils.py - HW4 Tool Calling Agent 共用模組

提供 API 呼叫、Prompt 建構、答案擷取、日誌記錄等共用功能。
"""

import requests
import json
import os
import sys
import time
import re
import random
import datetime
import threading

# ============================================================
# Configuration
# ============================================================
API_KEY_PATH = "key/api_key.txt"
MODEL_NAME = "google/gemma-4-31b-it"
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
REQUEST_INTERVAL = 1.6   # seconds between requests (40 RPM)
MAX_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.95
MAX_RETRIES = 3           # API 失敗時最多重試次數
RETRY_BACKOFF = 2.0       # 重試間隔倍數 (2s, 4s, 8s)


# ============================================================
# API Key Management
# ============================================================
def load_api_key(filepath=API_KEY_PATH):
    """從 txt 載入 NVIDIA NIM API Key"""
    if not os.path.exists(filepath):
        print(f"❌ 找不到 API Key 檔案！請建立 '{filepath}'")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        key = f.read().strip()
        if not key:
            print(f"❌ API Key 檔案 '{filepath}' 是空的！")
            sys.exit(1)
        return key


def load_api_keys(filepath=API_KEY_PATH):
    """從 txt 載入多把 NVIDIA NIM API Key（每行一把）"""
    if not os.path.exists(filepath):
        print(f"❌ 找不到 API Key 檔案！請建立 '{filepath}'")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        print(f"❌ API Key 檔案 '{filepath}' 是空的！")
        sys.exit(1)
    return keys


# ============================================================
# Logging
# ============================================================
def init_log(log_path):
    """初始化 log 檔案（JSONL 格式），回傳 log_path"""
    # 確保目錄存在
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_path


def append_log(log_path, record):
    """
    將一筆 API call 記錄寫入 JSONL log 檔。

    record 應包含:
      - timestamp: ISO 格式時間
      - id: sample id
      - system_prompt: (可選，第一筆記錄完整寫入)
      - user_prompt: 完整 user prompt
      - raw_response: API 原始回傳
      - extracted_answer: 擷取出的答案
      - ground_truth: (eval 模式才有)
      - is_correct: (eval 模式才有)
      - error: 錯誤訊息（如有）
      - retries: 重試次數
    """
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# API Calling (with retry)
# ============================================================
def query_gemma(system_prompt, user_prompt, api_key):
    """
    發送純文字請求給 Gemma-4-31B-IT，附帶自動重試機制。

    回傳: (response_text, retries, error_msg)
      - response_text: 模型回應文字，失敗時為 None
      - retries: 實際重試次數 (0 = 一次成功)
      - error_msg: 錯誤訊息，成功時為 None
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                INVOKE_URL, headers=headers, json=payload, timeout=240
            )
            response.raise_for_status()

            data = response.json()
            if (
                "choices" in data
                and isinstance(data["choices"], list)
                and len(data["choices"]) > 0
            ):
                content = (
                    data["choices"][0].get("message", {}).get("content", "")
                )
                if content and content.strip():
                    return content.strip(), attempt, None

            last_error = "API 回應格式不符預期或內容為空"

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "N/A"
            last_error = f"HTTP {status_code}: {str(e)}"
            # 429 Too Many Requests → 一定要等久一點
            if e.response and e.response.status_code == 429:
                wait_time = RETRY_BACKOFF * (2 ** attempt) * 2
                time.sleep(wait_time)
                continue
        except requests.exceptions.Timeout:
            last_error = "API 請求逾時 (240s)"
        except requests.exceptions.ConnectionError as e:
            last_error = f"連線錯誤: {str(e)}"
        except json.JSONDecodeError:
            last_error = "無法解析 API 傳回的 JSON"
        except Exception as e:
            last_error = f"未知例外: {str(e)}"

        # 等待後重試
        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_BACKOFF * (2 ** attempt)
            time.sleep(wait_time)

    return None, MAX_RETRIES, last_error


# ============================================================
# Prompt Construction
# ============================================================
SYSTEM_PROMPT = (
    "You are a tool-calling agent assistant. Your task is to select the most "
    "appropriate tool for the current step in a multi-step workflow.\n\n"
    "You will be given:\n"
    "1. The full task context showing the complete execution plan\n"
    "2. The current step that needs to be executed\n"
    "3. A list of candidate tools with their names, descriptions, arguments, "
    "and expected results\n\n"
    "Analyze the current step carefully and match it with the tool whose "
    "functionality best aligns with what the step requires. Pay attention to:\n"
    "- The tool's NAME and DESCRIPTION\n"
    "- Whether the tool's ARGUMENTS match the information available in the context\n"
    "- Whether the tool's RESULTS match what the current step is trying to achieve\n\n"
    "Think step by step, then output your final answer as: Answer: X\n"
    "where X is the letter of the best tool."
)


def format_tool_option(letter, tool):
    """格式化單一工具選項為文字描述"""
    tool_name = tool.get("name", "unknown")
    tool_desc = tool.get("description", "No description")

    lines = [f"{letter}. {tool_name}: {tool_desc}"]

    # Arguments
    if "arguments" in tool and "properties" in tool["arguments"]:
        props = tool["arguments"]["properties"]
        arg_parts = []
        for arg_name, arg_info in props.items():
            desc = arg_info.get("description", "")
            arg_type = arg_info.get("type", "")
            arg_parts.append(f"{arg_name} ({arg_type}): {desc}")
        lines.append("   Arguments:")
        for ap in arg_parts:
            lines.append(f"     - {ap}")

    # Results
    if "results" in tool and "properties" in tool["results"]:
        props = tool["results"]["properties"]
        res_parts = []
        for res_name, res_info in props.items():
            desc = res_info.get("description", "")
            res_parts.append(f"{res_name}: {desc}")
        lines.append("   Returns:")
        for rp in res_parts:
            lines.append(f"     - {rp}")

    return "\n".join(lines)


def build_tool_calling_prompt(sample):
    """
    根據一筆 sample 組裝 user prompt。

    回傳: (user_prompt, valid_option_letters)
      - user_prompt: 組裝好的 prompt 字串
      - valid_option_letters: ['A', 'B', ...] 合法選項列表
    """
    full_context = sample["full_context"]
    current_step = sample["current_step"]
    options = sample["options"]

    valid_option_letters = sorted(options.keys())

    # 構建候選工具列表
    tool_blocks = []
    for letter in valid_option_letters:
        tool_blocks.append(format_tool_option(letter, options[letter]))

    tools_text = "\n\n".join(tool_blocks)
    options_str = "/".join(valid_option_letters)

    user_prompt = (
        f"## Task Context\n{full_context}\n\n"
        f"## Current Step\n{current_step}\n\n"
        f"## Available Tools\n{tools_text}\n\n"
        f"## Instruction\n"
        f'Which tool should be used for the current step "{current_step}"?\n'
        f"Output your reasoning, then your final answer as: Answer: X\n"
        f"where X is one of ({options_str})."
    )

    return user_prompt, valid_option_letters


# ============================================================
# Structural-Only Prompt (Q2 Configuration)
# ============================================================
SYSTEM_PROMPT_STRUCTURAL = (
    "You are a tool-calling agent assistant. Your task is to select the most "
    "appropriate tool for the current step in a multi-step workflow.\n\n"
    "You will be given:\n"
    "1. The full task context showing the complete execution plan\n"
    "2. The current step that needs to be executed\n"
    "3. A list of candidate tools showing ONLY their parameter keys and "
    "data types (no names or descriptions are provided)\n\n"
    "Analyze the current step and determine which tool's parameter structure "
    "best matches the information and action required by the step.\n\n"
    "Think step by step, then output your final answer as: Answer: X\n"
    "where X is the letter of the best tool."
)


def format_tool_option_structural(letter, tool):
    """
    Structural-Only 格式化：只保留 parameter key + type，
    完全移除 tool name、tool description、argument descriptions、result descriptions。
    """
    lines = [f"{letter}."]

    # Arguments: 只列出 key: type
    if "arguments" in tool and "properties" in tool["arguments"]:
        props = tool["arguments"]["properties"]
        if props:
            lines.append("   Arguments:")
            for arg_name, arg_info in props.items():
                arg_type = arg_info.get("type", "string")
                lines.append(f"     - {arg_name}: {arg_type}")

    # Results: 只列出 key: type
    if "results" in tool and "properties" in tool["results"]:
        props = tool["results"]["properties"]
        if props:
            lines.append("   Returns:")
            for res_name, res_info in props.items():
                res_type = res_info.get("type", "string")
                lines.append(f"     - {res_name}: {res_type}")

    return "\n".join(lines)


def build_structural_only_prompt(sample):
    """
    Structural-Only 版本的 prompt 建構。
    與 build_tool_calling_prompt 結構相同，但工具資訊只保留結構。

    回傳: (user_prompt, valid_option_letters)
    """
    full_context = sample["full_context"]
    current_step = sample["current_step"]
    options = sample["options"]

    valid_option_letters = sorted(options.keys())

    tool_blocks = []
    for letter in valid_option_letters:
        tool_blocks.append(format_tool_option_structural(letter, options[letter]))

    tools_text = "\n\n".join(tool_blocks)
    options_str = "/".join(valid_option_letters)

    user_prompt = (
        f"## Task Context\n{full_context}\n\n"
        f"## Current Step\n{current_step}\n\n"
        f"## Available Tools\n{tools_text}\n\n"
        f"## Instruction\n"
        f'Which tool should be used for the current step "{current_step}"?\n'
        f"Output your reasoning, then your final answer as: Answer: X\n"
        f"where X is one of ({options_str})."
    )

    return user_prompt, valid_option_letters


# ============================================================
# Prompt Config Selector
# ============================================================
def get_prompt_builder(config="full_info"):
    """
    根據 config 回傳對應的 (system_prompt, build_fn)。

    config: 'full_info' | 'structural'
    """
    if config == "structural":
        return SYSTEM_PROMPT_STRUCTURAL, build_structural_only_prompt
    else:
        return SYSTEM_PROMPT, build_tool_calling_prompt


# ============================================================
# Answer Extraction
# ============================================================
def extract_answer(response_text, valid_options):
    """
    從模型回應中擷取答案字母。

    回傳: (answer, confidence)
      - answer: 擷取到的字母 (A-H)
      - confidence: 'high' | 'medium' | 'low' | 'fallback'
        表示擷取的可信度

    擷取策略（按優先順序）：
    1. "Answer: X" 明確格式 → high
    2. "The answer/tool/option is X" → high
    3. 回應末尾的獨立字母 → medium
    4. **X** 粗體字母 → medium
    5. 回應中第一個合法的獨立字母 → low
    6. 都找不到 → fallback (None)
    """
    if not response_text:
        return None, "fallback"

    text = response_text.strip()

    # Pattern 1: "Answer: X" (最標準的格式)
    match = re.search(r'[Aa]nswer\s*[:：]\s*\**([A-H])\**', text)
    if match and match.group(1) in valid_options:
        return match.group(1), "high"

    # Pattern 2: "The answer/tool/option/choice is X"
    match = re.search(
        r'(?:the\s+)?(?:best|correct|appropriate|most suitable)?\s*'
        r'(?:answer|tool|option|choice)\s+(?:is|would be|should be)\s+\**([A-H])\**',
        text, re.IGNORECASE,
    )
    if match and match.group(1).upper() in valid_options:
        return match.group(1).upper(), "high"

    # Pattern 3: 最後一行的獨立字母
    last_lines = text.strip().split("\n")
    for line in reversed(last_lines[-3:]):
        line_stripped = line.strip().strip("*").strip(".")
        if len(line_stripped) == 1 and line_stripped in valid_options:
            return line_stripped, "medium"

    # Pattern 4: **X** 粗體字母
    match = re.search(r'\*\*([A-H])\*\*', text)
    if match and match.group(1) in valid_options:
        return match.group(1), "medium"

    # Pattern 5: 回應中最後出現的合法獨立字母
    # （取最後一個，因為模型通常先推理再給答案）
    all_matches = re.findall(r'\b([A-H])\b', text)
    valid_matches = [m for m in all_matches if m in valid_options]
    if valid_matches:
        return valid_matches[-1], "low"

    # Fallback: 無法擷取
    return None, "fallback"


def heuristic_fallback(sample, valid_options):
    """
    當 API 完全失敗或無法擷取答案時，
    使用啟發式方法根據 current_step 與 tool name/description 做關鍵字匹配。

    這是最後的手段，準確率不高但比隨機好。
    """
    current_step = sample["current_step"].lower()
    options = sample["options"]

    best_score = -1
    best_letter = valid_options[0]  # 預設第一個

    for letter in valid_options:
        tool = options[letter]
        tool_name = tool.get("name", "").lower()
        tool_desc = tool.get("description", "").lower()

        score = 0

        # Tool name 中的每個 word 是否出現在 current_step 中
        name_words = re.findall(r'[a-z]+', tool_name)
        for word in name_words:
            if len(word) > 2 and word in current_step:
                score += 3

        # Tool description 中的關鍵字匹配
        desc_words = re.findall(r'[a-z]+', tool_desc)
        for word in desc_words:
            if len(word) > 3 and word in current_step:
                score += 1

        if score > best_score:
            best_score = score
            best_letter = letter

    return best_letter


# ============================================================
# Rate Limiter
# ============================================================
class RateLimiter:
    """簡易的 rate limiter，確保請求間隔不小於 interval 秒（thread-safe）"""

    def __init__(self, interval=REQUEST_INTERVAL):
        self.interval = interval
        self.last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self):
        """等待直到可以發送下一個請求（thread-safe）"""
        with self._lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.interval:
                wait_time = self.interval - elapsed
            else:
                wait_time = 0
            # 先更新時間再釋放鎖，防止多條 thread 同時通過
            self.last_request_time = time.time() + wait_time
        if wait_time > 0:
            time.sleep(wait_time)

    def tick(self):
        """記錄本次請求時間"""
        with self._lock:
            self.last_request_time = time.time()


class MultiKeyRateLimiter:
    """
    多 API Key 輪流使用的 rate limiter。
    每把 key 各自維護冷卻時間，每次取「等待時間最短」的 key，
    藉此將有效 throughput 提升至 N 倍（N = key 數量）。
    """

    def __init__(self, api_keys, interval=REQUEST_INTERVAL):
        self.api_keys = api_keys
        self.interval = interval
        self.last_request_times = [0.0] * len(api_keys)
        self._current_idx = 0

    def get_next_key(self):
        """
        回傳下一把可用的 API key（等待時間最短的那把）。
        若仍需等待，會自動 sleep。
        回傳: (api_key, key_index)
        """
        now = time.time()
        best_idx = 0
        min_wait = float("inf")

        for i in range(len(self.api_keys)):
            elapsed = now - self.last_request_times[i]
            wait = max(0.0, self.interval - elapsed)
            if wait < min_wait:
                min_wait = wait
                best_idx = i

        if min_wait > 0:
            time.sleep(min_wait)

        self._current_idx = best_idx
        return self.api_keys[best_idx], best_idx

    def tick(self):
        """記錄目前使用的 key 的請求時間"""
        self.last_request_times[self._current_idx] = time.time()


# ============================================================
# Data Loading
# ============================================================
def load_jsonl(filepath):
    """載入 JSONL 檔案，回傳 list of dict"""
    if not os.path.exists(filepath):
        print(f"❌ 找不到檔案: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def now_iso():
    """取得當前時間的 ISO 格式字串"""
    return datetime.datetime.now().isoformat()
