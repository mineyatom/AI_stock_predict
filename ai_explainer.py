import os
import time
from google import genai


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


DISCLAIMER = (
    "本內容僅為模型統計推論，"
    "不構成投資建議。"
)


def generate_ai_analysis(result):

    try:
        confidence = float(result["confidence"])
        direction = str(result["direction"]).strip()
        price_range = result["price_range"]

        top_features = result.get(
            "top_features",
            []
        )

        feature_text = "、".join(
            top_features
        )

        positive_factors = result.get(
            "positive_factors",
            []
        )

        negative_factors = result.get(
            "negative_factors",
            []
        )

        positive_text = "、".join(
            positive_factors
        )   

        negative_text = "、".join(
            negative_factors
        )

        if 0 <= confidence <= 1:
            confidence = confidence * 100

        if confidence >= 70:
            confidence_level = "高信心"
        elif confidence >= 55:
            confidence_level = "中等信心"
        else:
            confidence_level = "低信心"

        if direction == "上漲":
            if confidence >= 70:
                model_view = "模型偏向上漲，短線動能相對較強"
            elif confidence >= 55:
                model_view = "模型略偏上漲，但方向仍有不確定性"
            else:
                model_view = "模型雖偏向上漲，但判斷力道有限"

        elif direction == "下跌":
            if confidence >= 70:
                model_view = "模型偏向下跌，短線壓力相對較大"
            elif confidence >= 55:
                model_view = "模型略偏下跌，但方向仍有不確定性"
            else:
                model_view = "模型雖偏向下跌，但判斷力道有限"

        else:
            model_view = "模型方向不明確，短線仍偏觀望"

        fallback_text = (
            f"{model_view}。"
        )

        if feature_text:
            fallback_text += (
                f"模型整體較重視 {feature_text} "
                "等市場因素。"
            )

        fallback_text += (
            f"預測區間為 {price_range}。"
        )

        prompt = f"""
你是一位金融 AI 助手。

方向：
{direction}

信心等級：
{confidence_level}

模型觀點：
{model_view}

模型重要特徵：
{feature_text}

預測區間：
{price_range}

本次正向因素：
{positive_text}

本次負向因素：
{negative_text}

規則：

1. 僅根據提供資料說明
2. 不得推測新聞
3. 不得推測法人動向
4. 不得推測資金流向
5. 不得新增模型未提供資訊
6. 模型重要特徵代表模型整體較重視的市場因素
7. 不得表示這些特徵一定是本次預測的直接原因
8. 使用繁體中文
9. 控制在 80 字內
10. 不要加入免責聲明
11. 本次正向因素與負向因素來自 SHAP 單筆解釋
12. 可說明正負向因素形成支撐或壓力，但不得說成絕對因果
13. 最多 2 句
14. 不要同時列出太多特徵，正向與負向各最多提 1~2 個
15. 不要重複「模型整體較重視」與「本次預測」兩種解釋太多次
"""

        response = None

        for attempt in range(3):

            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                break

            except Exception as e:

                if attempt == 2:
                    raise e

                print(
                    f"Gemini 暫時忙碌，重試第 {attempt + 1} 次..."
                )

                time.sleep(2)

        ai_text = response.text.strip()

        if not ai_text:
            ai_text = fallback_text

        return f"{ai_text}\n\n{DISCLAIMER}"

    except Exception as e:
        print(f"Gemini 解讀失敗：{e}")

        return (
            f"{fallback_text}\n\n{DISCLAIMER}"
            if "fallback_text" in locals()
            else (
                "目前無法產生 AI 解讀。\n\n"
                f"{DISCLAIMER}"
            )
        )