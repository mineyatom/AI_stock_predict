def build_market_summary_prompt(
    market_data,
    prediction_summary: dict
) -> str:
    """
    建立 AI 市場分析 Prompt
    """

    market_text_lines = []

    if isinstance(market_data, list):
        for item in market_data:
            market_text_lines.append(
                f"- {item.get('name', '未知市場')}："
                f"{item.get('price', '無資料')}，"
                f"{item.get('change', '無資料')}"
            )

    elif isinstance(market_data, dict):
        for key, value in market_data.items():
            market_text_lines.append(
                f"- {key}：{value}"
            )

    else:
        market_text_lines.append("- 市場資料：無資料")

    market_text = "\n".join(market_text_lines)

    total_stocks = prediction_summary.get("total_stocks", 0)
    up_count = prediction_summary.get("up_count", 0)
    down_count = prediction_summary.get("down_count", 0)
    high_confidence_count = prediction_summary.get(
        "high_confidence_count",
        0
    )
    medium_confidence_count = prediction_summary.get(
        "medium_confidence_count",
        0
    )
    low_confidence_count = prediction_summary.get(
        "low_confidence_count",
        0
    )

    stock_lines = []

    for stock in prediction_summary.get("stocks", []):
        stock_lines.append(
            f"""
股票：{stock.get("stock_name", "")}（{stock.get("stock_id", "")}）
模型預測方向：偏向{stock.get("prediction", "")}
模型可信度：{stock.get("confidence_level", "")}
"""
        )

    stock_text = "\n".join(stock_lines)

    prompt = f"""
你是 AI 股票模型分析助手。

你的工作是根據 Python 已完成分析的市場資料與模型結果，
整理成自然、專業且簡潔的市場分析。

========================
【硬性規則（必須遵守）】
========================

1. 只能根據提供資料整理內容。
2. 不得加入任何外部新聞。
3. 不得自行推測未提供資訊。
4. 不得描述未來市場走勢。
5. 不得提供任何投資建議。
6. 若提及個股方向，只能使用：
   ・模型預測方向偏向上漲
   ・模型預測方向偏向下跌
7. 若模型訊號不一致，請使用：
   「模型整體訊號仍存在分歧」
8. 若資料不足，請使用：
   「模型目前可提供資訊有限。」
9. 不得自行補充任何內容。

========================
【禁止使用詞語】
========================

禁止使用以下詞語及其近義詞：

建議
留意
觀察
推薦
值得
應該
宜
可考慮
布局
操作
買進
賣出
加碼
減碼
短線
長線

========================
【寫作規則】
========================

1. 使用繁體中文。
2. 使用金融分析報告口吻。
3. 字數控制於 120~180 字。
4. 不要列點。
5. 不要加標題。
6. 不要加入前言或結尾。
7. 直接輸出分析內容。
8. 以 4~6 句完成。
9. 不要逐字重複市場資料。
10. 不要完整重述所有數值。
11. 請整理重點，而不是逐項念資料。

========================
【內容順序】
========================

請依照以下順序撰寫：

① 市場概況

② 模型整體分析

③ 模型預測重點

========================
【今日市場資料】
========================

{market_text}

========================
【今日模型統計】
========================

總股票數：{total_stocks}

模型預測方向偏向上漲：{up_count}

模型預測方向偏向下跌：{down_count}

高可信度：{high_confidence_count}

中可信度：{medium_confidence_count}

低可信度：{low_confidence_count}

========================
【模型預測個股】
========================

{stock_text}
"""

    return prompt.strip()