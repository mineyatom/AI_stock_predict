import requests

from prompt_builder import build_prediction_prompt


OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"


def analyze_prediction_with_ollama(prediction_data):
    """
    使用 Ollama 針對單筆預測結果產生 AI 解讀。

    Business Logic：
        confidence_manager.py

    Prompt：
        prompt_builder.py

    LLM Client：
        ollama_analyzer.py
    """

    try:

        prompt = build_prediction_prompt(
            prediction_data
        )

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

        print(f"Ollama 分析失敗：{e}")

        return (
            "AI 解讀暫時無法產生。\n"
            "本內容僅供模型分析參考，不構成投資建議。"
        )