import requests

from feature_description import explain_feature_list


OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"


def analyze_prediction_with_ollama(prediction_data):
    """
    使用 Ollama 針對單筆預測結果產生 AI 解讀
    注意：Ollama 不負責預測，只負責解釋既有模型結果
    """

    try:
        positive_factors = explain_feature_list(
        prediction_data.get("positive_factors")
        )

        negative_factors = explain_feature_list(
        prediction_data.get("negative_factors")
        )

        positive_factors_text = "、".join(positive_factors)

        negative_factors_text = "、".join(negative_factors)

        prompt = f"""
你是台股 AI 模型分析助手。

你的任務是解釋 XGBoost 模型輸出的預測結果。
你不是投資顧問，也不能自行預測股價。

請根據以下資料，用繁體中文產生一段專業但容易理解的分析。

輸出規則：
1. 只能解釋模型結果，不要新增資料。
2. 不要使用「建議買進、賣出、持有」等投資建議。
3. 不要保證結果會發生。
4. 文字控制在 120 到 180 字。
5. 語氣要像金融 Dashboard 的 AI 分析說明。
6. 最後一定要加上：
「本內容僅供模型分析參考，不構成投資建議。」

模型資料：
股票：{prediction_data.get("stock_name")}（{prediction_data.get("stock_id")}）
預測方向：{prediction_data.get("direction")}
信心值：{prediction_data.get("confidence")}%
上漲機率：{prediction_data.get("up_probability")}%
下跌機率：{prediction_data.get("down_probability")}%
預測區間：{prediction_data.get("price_range")}


SHAP 正向因素：
{positive_factors_text}

SHAP 負向因素：
{negative_factors_text}

請按照以下格式輸出：

本次模型預測股票名稱偏向預測方向，信心值為信心值%。
主要支撐因素包含「正向因素」，代表模型認為這些變數對預測方向形成支撐。
不過「負向因素」仍對結果造成壓力，顯示短線仍存在不確定性。
本內容僅供模型分析參考，不構成投資建議。
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