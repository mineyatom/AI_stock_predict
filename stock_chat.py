import requests

from chat_prompt_builder import build_chat_prompt


OLLAMA_API_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = "qwen3:8b"


def chat_with_model(
    question: str,
    prediction_data: dict
) -> str:
    """
    AI 模型助手

    Parameters
    ----------
    question
        使用者問題

    prediction_data
        predict_stock() 回傳資料

    Returns
    -------
    str
        AI 回答
    """

    try:

        prompt = build_chat_prompt(
            question,
            prediction_data
        )

        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        return data.get(
            "response",
            "AI 無法回答。"
        ).strip()

    except Exception as e:

        print("AI 模型助手錯誤：", e)

        return (
            "AI 模型助手目前無法使用。\n"
            "請稍後再試。"
        )