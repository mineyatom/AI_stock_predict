"""
confidence_manager.py

負責管理模型信心值相關邏輯。

Business Logic：
- 信心等級
- AI 描述語氣
"""


def get_confidence_level(confidence):
    """
    回傳模型可信度等級
    """

    confidence = float(confidence)

    if confidence >= 75:
        return "高可信度"

    elif confidence >= 65:
        return "偏高可信度"

    elif confidence >= 55:
        return "中等可信度"

    else:
        return "低可信度"


def get_confidence_context(confidence):
    """
    根據信心值回傳 AI 分析語氣。

    這些內容提供給 Prompt Builder 使用，
    Ollama 只負責自然語言潤稿。
    """

    confidence = float(confidence)

    if confidence >= 75:

        return {

            "level": "高可信度",

            "tone":
                "多數模型訊號方向一致，模型對目前判斷具有較高信心。",

            "uncertainty":
                "請不要強調模型存在明顯不確定性，只需提醒結果仍屬機率預測。",

        }

    elif confidence >= 65:

        return {

            "level": "偏高可信度",

            "tone":
                "多數訊號支持目前方向，但仍有少部分訊號存在差異。",

            "uncertainty":
                "可簡單說明仍保留少量不確定性。",

        }

    elif confidence >= 55:

        return {

            "level": "中等可信度",

            "tone":
                "正向與負向訊號同時存在，模型綜合分析後形成目前判斷。",

            "uncertainty":
                "可說明模型仍存在一定程度的不確定性。",

        }

    else:

        return {

            "level": "低可信度",

            "tone":
                "目前多項訊號分歧明顯，因此模型判斷較為保守。",

            "uncertainty":
                "請明確說明模型信心偏低，但不要加入任何投資建議。",

        }