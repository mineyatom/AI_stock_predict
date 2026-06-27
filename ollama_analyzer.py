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

        positive_factors_text = "\n".join(
            f"• {item}" for item in positive_factors
        )

        negative_factors_text = "\n".join(
            f"• {item}" for item in negative_factors
        )

        prompt = f"""
你是一位 AI 股票模型分析師。

你的工作只有一個：
解釋 XGBoost 模型輸出的結果。

請務必遵守：

1. 不要自行預測股價。
2. 不要提供買進、賣出或持有建議。
3. 不要使用：
   - 支撐上漲
   - 支撐下跌
   - 看多
   - 看空
   - 利多
   - 利空
4. 只能描述模型判斷依據。
5. 使用繁體中文。
6. 回答控制在 180 字內。
7. 最後一定加入：
「本內容僅供模型分析參考，不構成投資建議。」

模型資料

股票：
{prediction_data.get("stock_name")}（{prediction_data.get("stock_id")}）

模型預測方向：
{prediction_data.get("direction")}

模型信心值：
{prediction_data.get("confidence")}%

上漲機率：
{prediction_data.get("up_probability")}%

下跌機率：
{prediction_data.get("down_probability")}%

預測區間：
{prediction_data.get("price_range")}

提高模型信心的因素：

{positive_factors_text}

降低模型信心的因素：

{negative_factors_text}

請只輸出一段自然流暢的模型摘要。

摘要需包含：

1. 模型預測方向。
2. 信心值高低。
3. 哪些因素提高模型信心。
4. 哪些因素降低模型信心。

不要使用：

- 標題
- 條列
- Markdown
- 【模型預測】
- 【主要影響因素】
- 【模型說明】

請以一般文章方式回答，控制在 3~4 句。

最後一定加上：

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