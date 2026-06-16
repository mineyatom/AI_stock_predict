import os
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

        # 防止 0~1 機率格式誤傳
        if confidence <= 1:
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

        prompt = f"""
你是金融平台的文字潤稿助手。

請根據已分類好的模型摘要，
輸出一句自然、專業、保守的繁體中文解讀。

限制：
1. 只能潤稿，不得重新判斷方向
2. 不得提及法人、新聞、資金流向、市場情緒
3. 不得重複百分比數字
4. 使用「可能、偏向、顯示」等機率語氣
5. 使用「預測區間」一詞
6. 控制在 50 字以內
7. 不要條列式
8. 不要加入免責聲明

模型摘要：
---
方向：{direction}
信心等級：{confidence_level}
模型觀察：{model_view}
預測區間：{price_range}
---
請直接輸出一句解讀。
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        ai_text = response.text.strip()

        if not ai_text:
            ai_text = "模型已完成預測，但目前無法產生完整文字解讀。"

        return f"{ai_text}\n\n{DISCLAIMER}"

    except Exception as e:
        print(f"Gemini 解讀失敗：{e}")

        return (
            "目前無法產生 AI 解讀。\n\n"
            f"{DISCLAIMER}"
        )