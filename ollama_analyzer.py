import requests


OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"


def analyze_prediction_with_ollama(prediction_data):
    """
    使用 Ollama 針對單筆預測結果產生 AI 解讀
    注意：Ollama 不負責預測，只負責解釋既有模型結果
    """

    try:
        prompt = f"""
你是台股 AI 模型分析助手。

請根據以下 XGBoost 模型預測結果，產生一段繁體中文分析。

重要規則：
1. 不要自己預測股價。
2. 不要給買進、賣出、持有建議。
3. 只能解釋模型結果。
4. 最後一定要加上：本內容僅供模型分析參考，不構成投資建議。

資料如下：
股票：{prediction_data.get("stock_name")}（{prediction_data.get("stock_id")}）
預測方向：{prediction_data.get("direction")}
信心值：{prediction_data.get("confidence")}%
上漲機率：{prediction_data.get("up_probability")}%
下跌機率：{prediction_data.get("down_probability")}%
預測區間：{prediction_data.get("price_range")}
正向因素：{prediction_data.get("positive_factors")}
負向因素：{prediction_data.get("negative_factors")}

請用 3 到 5 句話說明。
"""

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
            "AI 解讀暫時無法產生。"
            "本內容僅供模型分析參考，不構成投資建議。"
        )