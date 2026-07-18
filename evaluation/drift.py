from evaluation.health import (
    get_overall_accuracy,
    get_recent_accuracy,
)


def calculate_drift() -> dict:
    """
    Model Drift Detection

    比較：
        整體歷史勝率
            vs
        最近7日勝率
    """

    overall = get_overall_accuracy()
    recent7 = get_recent_accuracy(7)

    drift = round(
        recent7["accuracy"] - overall["accuracy"],
        2
    )

    abs_drift = abs(drift)

    if abs_drift < 3:
        level = "正常"
        icon = "🟢"
        color = "green"

    elif abs_drift < 8:
        level = "注意"
        icon = "🟡"
        color = "orange"

    else:
        level = "異常"
        icon = "🔴"
        color = "red"

    return {
        "drift": drift,
        "abs_drift": abs_drift,
        "level": level,
        "icon": icon,
        "color": color,
        "overall": overall["accuracy"],
        "recent7": recent7["accuracy"],
    }


def get_drift_message(level: str) -> str:
    """
    AI Report 使用
    """

    messages = {
        "正常":
            "近期模型與歷史表現一致，尚未觀察到明顯模型漂移。",

        "注意":
            "近期模型表現開始偏離歷史平均，建議持續觀察模型狀態。",

        "異常":
            "近期模型已出現明顯漂移，建議重新訓練模型或重新檢查特徵工程。"
    }

    return messages[level]


def need_retrain(level: str) -> bool:
    """
    是否建議重新訓練
    """

    return level == "異常"