from datetime import datetime, timedelta

from prediction_repository import (
    get_validated_predictions_from_db,
)


def get_recent_accuracy(days: int) -> dict:
    """
    計算最近 N 天模型勝率
    """

    predictions = get_validated_predictions_from_db()

    cutoff = (
        datetime.today() - timedelta(days=days)
    ).date()

    recent = []

    for item in predictions:

        predict_date = datetime.strptime(
            str(item["predict_date"]),
            "%Y-%m-%d"
        ).date()

        if predict_date >= cutoff:
            recent.append(item)

    total = len(recent)

    if total == 0:
        return {
            "accuracy": 0,
            "correct": 0,
            "total": 0,
        }

    correct = sum(
        1
        for item in recent
        if item["is_correct"] == "正確"
    )

    return {
        "accuracy": round(correct / total * 100, 2),
        "correct": correct,
        "total": total,
    }


def get_overall_accuracy() -> dict:
    """
    整體歷史勝率
    """

    predictions = get_validated_predictions_from_db()

    total = len(predictions)

    if total == 0:
        return {
            "accuracy": 0,
            "correct": 0,
            "total": 0,
        }

    correct = sum(
        1
        for item in predictions
        if item["is_correct"] == "正確"
    )

    return {
        "accuracy": round(correct / total * 100, 2),
        "correct": correct,
        "total": total,
    }


def calculate_health_score() -> dict:
    """
    計算 Model Health Score
    """

    recent7 = get_recent_accuracy(7)
    recent30 = get_recent_accuracy(30)
    overall = get_overall_accuracy()

    sample_score = (
        min(overall["total"], 500) / 500 * 100
    )

    health_score = round(
        recent7["accuracy"] * 0.4
        + recent30["accuracy"] * 0.3
        + overall["accuracy"] * 0.2
        + sample_score * 0.1,
        1,
    )

    return {
        "score": health_score,
        "recent7": recent7,
        "recent30": recent30,
        "overall": overall,
    }


def get_health_status(score: float) -> str:

    if score >= 80:
        return "🟢 良好"

    if score >= 60:
        return "🟡 觀察"

    return "🔴 建議重新訓練"


def get_health_advice(score: float) -> str:

    if score >= 80:
        return "模型近期表現穩定，可維持目前模型版本。"

    if score >= 60:
        return "模型整體正常，建議持續觀察近期勝率變化。"

    return "近期模型表現下降，建議重新訓練或重新檢查特徵工程。"