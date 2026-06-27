"""
prompt_builder.py

負責建立 LLM Prompt。

這裡只處理：
- Prompt 組裝
- LLM 指令格式

不負責：
- 信心值判斷
- API 呼叫
- 模型預測
"""


from feature_description import explain_feature_list
from confidence_manager import get_confidence_context


def build_prediction_prompt(prediction_data):
    """
    建立單筆股票預測 AI 解讀 Prompt
    """

    confidence = prediction_data.get(
        "confidence",
        0
    )

    confidence_context = get_confidence_context(
        confidence
    )

    positive_factors = explain_feature_list(
        prediction_data.get(
            "positive_factors",
            []
        )
    )

    negative_factors = explain_feature_list(
        prediction_data.get(
            "negative_factors",
            []
        )
    )

    positive_factors_text = "\n".join(
        f"• {item}"
        for item in positive_factors
    )

    negative_factors_text = "\n".join(
        f"• {item}"
        for item in negative_factors
    )

    prompt = f"""
你是一位 AI 股票模型分析師。

你的任務只有一個：
解釋 XGBoost 模型輸出的結果。

請務必遵守：

1. 不要自行預測股價。
2. 不要提供買進、賣出、持有等投資建議。
3. 不要使用：
   - 看多
   - 看空
   - 利多
   - 利空
   - 支撐上漲
   - 支撐下跌
4. 只能解釋模型判斷依據。
5. 使用繁體中文。
6. 回答控制在 100～150 字。
7. 不要使用 Markdown、標題、條列或特殊格式。
8. 最後一定加入：
本內容僅供模型分析參考，不構成投資建議。

====================
模型資料
====================

股票：
{prediction_data.get("stock_name")}（{prediction_data.get("stock_id")}）

模型預測方向：
{prediction_data.get("direction")}

模型信心值：
{prediction_data.get("confidence")}%

模型可信度：
{confidence_context["level"]}

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

====================
語氣控制
====================

模型語氣設定：
{confidence_context["tone"]}

不確定性描述規則：
{confidence_context["uncertainty"]}

====================
回答規則
====================

請以自然流暢的繁體中文回答，不要使用標題、條列、Markdown 或特殊格式。

回答內容請依照以下順序：

1. 使用「模型預測方向偏向上漲」或「模型預測方向偏向下跌」，不要使用「股價將上漲／股價將下跌」等肯定語句。

2. 使用提供的模型可信度，不要自行判斷可信度等級，也不要自行修改可信度描述。

3. 說明模型是根據價格、技術指標、法人籌碼與市場訊號等多項資訊綜合分析後形成目前判斷。

4. 請依照提供的語氣設定與不確定性描述規則，自然說明模型目前的判斷，不要固定提及不確定性。

5. 不要逐項列出提高模型信心與降低模型信心的因素，因為畫面下方已經會顯示。

6. 不要重複信心值、機率、預測方向或因素名稱。

7. 不要加入任何投資建議，例如：
- 建議
- 可留意
- 可觀察
- 可考慮
- 謹慎評估
- 搭配其他指標

8. 不要使用過於肯定的語氣，例如：
- 必定
- 一定
- 勢必
- 必然

9. 請保持金融分析平台的專業、客觀語氣，以「模型解讀」為核心，而非投資評論。

回答長度控制在 100～150 字。

最後固定輸出：

本內容僅供模型分析參考，不構成投資建議。
"""

    return prompt