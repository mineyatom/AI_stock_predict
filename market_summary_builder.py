def build_market_summary_prompt(
    market_data,
    prediction_summary: dict
) -> str:
    """
    建立 AI 每日市場摘要 Prompt
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
信心值：{stock.get("confidence", "")}%
可信度：{stock.get("confidence_level", "")}
"""
        )

    stock_text = "\n".join(stock_lines)

    prompt = f"""
你是 AI 股票模型摘要助手。

請根據以下「模型與市場資料」產生一段每日市場摘要。

重要規則：
1. 不要提供任何投資建議。
2. 不要使用「建議」、「留意」、「觀察」、「謹慎評估」等字詞。
3. 不要寫「股價將上漲」或「股價將下跌」。
4. 必須使用「模型預測方向偏向上漲」或「模型預測方向偏向下跌」這種表達。
5. 只能根據提供的資料整理，不要補充外部新聞或自行推測。
6. 字數控制在 100 到 160 字。
7. 使用繁體中文。

今日市場資料：
{market_text}

今日模型預測統計：
- 總股票數：{total_stocks}
- 偏向上漲：{up_count}
- 偏向下跌：{down_count}
- 高可信度：{high_confidence_count}
- 中可信度：{medium_confidence_count}
- 低可信度：{low_confidence_count}

個股模型結果：
{stock_text}

請以金融分析師的口吻，
整理今日市場觀點。

不要列點。

不要重複市場數據。

請以 4~6 句完成。

不要提供投資建議。

不要加標題。
"""

    return prompt.strip()