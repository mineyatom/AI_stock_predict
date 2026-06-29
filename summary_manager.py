

import os
import pandas as pd

from confidence_manager import get_confidence_level


LOG_FILE = "prediction_log.csv"


def get_today_prediction_summary() -> dict:
    """
    整理今日預測摘要

    來源：
    prediction_log.csv

    責任：
    - 統計今日預測總數
    - 統計上漲 / 下跌數量
    - 統計高 / 中 / 低可信度數量
    - 整理今日個股預測清單
    """

    default_summary = {
        "total_stocks": 0,
        "up_count": 0,
        "down_count": 0,
        "high_confidence_count": 0,
        "medium_confidence_count": 0,
        "low_confidence_count": 0,
        "stocks": []
    }

    if not os.path.exists(LOG_FILE):
        return default_summary

    df = pd.read_csv(LOG_FILE)

    if df.empty:
        return default_summary

    if "預測日期" not in df.columns:
        return default_summary

    df["預測日期"] = pd.to_datetime(
        df["預測日期"],
        errors="coerce"
    ).dt.date

    today = pd.Timestamp.today().date()

    today_df = df[
        df["預測日期"] == today
    ]

    if today_df.empty:
        return default_summary

    for _, row in today_df.iterrows():

        prediction = str(
            row.get("預測結果", "")
        )

        confidence = row.get(
            "信心值",
            0
        )

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0

        confidence_level = get_confidence_level(
            confidence
        )

        if prediction == "上漲":
            default_summary["up_count"] += 1

        elif prediction == "下跌":
            default_summary["down_count"] += 1

        if confidence_level == "高可信度":
            default_summary["high_confidence_count"] += 1

        elif confidence_level == "中可信度":
            default_summary["medium_confidence_count"] += 1

        else:
            default_summary["low_confidence_count"] += 1

        default_summary["stocks"].append({
            "stock_id": row.get("股票代號", ""),
            "stock_name": row.get("股票名稱", ""),
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "confidence_level": confidence_level
        })

    default_summary["total_stocks"] = len(today_df)

    return default_summary


def get_market_signal(
    prediction_summary: dict
):
    """
    根據今日預測方向統計，判斷市場訊號
    """

    up_count = prediction_summary.get("up_count", 0)
    down_count = prediction_summary.get("down_count", 0)

    if up_count >= down_count + 2:
        return "🟢 偏多", "signal-bull"

    if down_count >= up_count + 2:
        return "🔴 偏空", "signal-bear"

    return "🟡 中性", "signal-neutral"


def get_model_confidence(
    prediction_summary: dict
):
    """
    根據今日高可信度比例，判斷整體模型可信度
    """

    total = prediction_summary.get("total_stocks", 0)

    if total == 0:
        return "🟡 中可信度", "signal-medium"

    high_count = prediction_summary.get(
        "high_confidence_count",
        0
    )

    high_ratio = high_count / total

    if high_ratio >= 0.4:
        return "🟢 高可信度", "signal-bull"

    if high_ratio >= 0.2:
        return "🟡 中可信度", "signal-medium"

    return "🔴 低可信度", "signal-bear"