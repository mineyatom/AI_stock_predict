from evaluation.health import get_recent_accuracy


def calculate_trend() -> dict:
    """
    比較最近 7 天與最近 30 天模型表現
    """

    recent7 = get_recent_accuracy(7)
    recent30 = get_recent_accuracy(30)

    trend = round(
        recent7["accuracy"] - recent30["accuracy"],
        2
    )

    if trend >= 5:
        direction = "明顯改善"
        icon = "📈"

    elif trend >= 1:
        direction = "略為改善"
        icon = "↗️"

    elif trend > -1:
        direction = "維持穩定"
        icon = "➡️"

    elif trend > -5:
        direction = "略為下降"
        icon = "↘️"

    else:
        direction = "明顯下降"
        icon = "📉"

    return {
        "trend": trend,
        "direction": direction,
        "icon": icon,
        "recent7": recent7["accuracy"],
        "recent30": recent30["accuracy"],
    }


def get_trend_message(trend: float) -> str:
    """
    AI 報告使用的描述文字
    """

    if trend >= 5:
        return "近期模型表現明顯優於過去 30 天，模型仍持續改善。"

    if trend >= 1:
        return "近期模型表現略有提升，預測能力維持向上。"

    if trend > -1:
        return "近期模型表現與長期平均相近，整體維持穩定。"

    if trend > -5:
        return "近期模型勝率略低於長期平均，建議持續觀察。"

    return "近期模型表現明顯下降，可能開始產生模型老化現象。"