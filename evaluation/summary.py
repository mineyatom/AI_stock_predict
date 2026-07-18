from evaluation.health import (
    calculate_health_score,
    get_health_status,
    get_health_advice,
)

from evaluation.trend import (
    calculate_trend,
    get_trend_message,
)

from evaluation.drift import (
    calculate_drift,
    get_drift_message,
)


def generate_ai_summary() -> str:
    """
    AI Evaluation Report
    根據目前模型狀態自動產生分析內容
    """

    health = calculate_health_score()
    trend = calculate_trend()
    drift = calculate_drift()

    score = health["score"]

    status = get_health_status(score)
    advice = get_health_advice(score)

    trend_msg = get_trend_message(
        trend["trend"]
    )

    drift_msg = get_drift_message(
        drift["level"]
    )

    if score >= 80:
        overall = (
            "模型目前維持健康狀態，"
            "整體預測能力穩定，"
            "可持續使用目前模型。"
        )

    elif score >= 60:
        overall = (
            "模型整體維持正常，"
            "近期仍具備一定預測能力，"
            "建議持續觀察近期表現。"
        )

    else:
        overall = (
            "模型近期表現開始下降，"
            "建議安排重新訓練，"
            "並重新檢查特徵工程與資料品質。"
        )

    report = f"""
AI 模型評估報告

━━━━━━━━━━━━━━━━━━━━

🩺 模型健康狀態

{status}

Health Score：{score:.2f}%

{advice}

━━━━━━━━━━━━━━━━━━━━

📈 模型趨勢分析

{trend['icon']} {trend['direction']}

近7日：
{trend['recent7']:.2f}%

近30日：
{trend['recent30']:.2f}%

差異：
{trend['trend']:+.2f}%

{trend_msg}

━━━━━━━━━━━━━━━━━━━━

🛰 模型穩定度

{drift['icon']} {drift['level']}

整體勝率：
{drift['overall']:.2f}%

近7日：
{drift['recent7']:.2f}%

Drift：
{drift['drift']:+.2f}%

{drift_msg}

━━━━━━━━━━━━━━━━━━━━

📋 整體評估

{overall}

本分析結果僅供模型驗證與研究用途，
不構成任何投資建議。
"""

    return report.strip()