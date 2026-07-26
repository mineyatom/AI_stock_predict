"""
confidence_manager.py

負責管理模型信心值相關邏輯。

Business Logic:
- 信心等級
- AI 描述語氣

V11.5 Update:
根據 Benchmark Confidence Threshold Analysis 調整門檻：

80%以上:
    極高可信度

70~80%:
    高可信度

60~70%:
    一般可信度

60%以下:
    低可信度

Ollama 只負責自然語言潤稿，
不負責判斷模型可信度。
"""



def get_confidence_level(confidence):
    """
    回傳模型可信度等級
    """

    confidence = float(confidence)


    if confidence >= 80:

        return "極高可信度"


    elif confidence >= 70:

        return "高可信度"


    elif confidence >= 60:

        return "一般可信度"


    else:

        return "低可信度"





def get_confidence_context(confidence):
    """
    根據信心值回傳 AI 分析語氣。

    提供 Prompt Builder 使用。

    Ollama 僅負責自然語言生成，
    不進行模型判斷。
    """

    confidence = float(confidence)



    if confidence >= 80:

        return {

            "level":

                "極高可信度",


            "tone":

                "模型信心度較高，目前多數模型訊號支持相同方向。",


            "uncertainty":

                "仍需說明結果屬於機率預測，不代表未來一定發生。",

        }



    elif confidence >= 70:

        return {

            "level":

                "高可信度",


            "tone":

                "多數模型訊號支持目前方向，模型具有較高方向判斷信心。",


            "uncertainty":

                "仍可能受到市場環境變化影響，需保留一定不確定性。",

        }



    elif confidence >= 60:

        return {

            "level":

                "一般可信度",


            "tone":

                "模型綜合多項特徵後形成目前判斷，但部分訊號仍存在差異。",


            "uncertainty":

                "模型仍存在一定程度的不確定性，需要觀察後續市場變化。",

        }



    else:

        return {

            "level":

                "低可信度",


            "tone":

                "目前模型訊號分歧較大，因此判斷較為保守。",


            "uncertainty":

                "請明確說明模型信心偏低，避免過度解讀預測結果。",

        }