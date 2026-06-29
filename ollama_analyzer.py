import requests

from prompt_builder import build_prediction_prompt


OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"


def call_ollama(prompt: str) -> str:
    """
    Ollama 共用呼叫函式

    責任：
    - 只負責把 prompt 送給 Ollama
    - 不負責組 Prompt
    - 不負責判斷可信度
    - 不負責 Business Logic
    """

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        return data.get(
            "response",
            "Ollama 未回傳分析內容。"
        ).strip()

    except Exception as e:
        print(f"Ollama 呼叫失敗：{e}")

        return (
            "AI 內容暫時無法產生。\n"
            "本內容僅供模型分析參考，不構成投資建議。"
        )


def analyze_prediction_with_ollama(prediction_data: dict) -> str:
    """
    使用 Ollama 針對單筆預測結果產生 AI 解讀。
    """

    prompt = build_prediction_prompt(
        prediction_data
    )

    return call_ollama(prompt)