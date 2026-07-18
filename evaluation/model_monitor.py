from evaluation.health import (
    calculate_health_score,
    get_health_status,
    get_health_advice,
)

from evaluation.trend import (
    calculate_trend,
)

from evaluation.drift import (
    calculate_drift,
)

from evaluation.summary import (
    generate_ai_summary,
)


def get_model_monitor() -> dict:
    """
    Evaluation 模組統一入口

    app.py 只需要：

        monitor = get_model_monitor()

    不需要知道：
        Health
        Trend
        Drift
        Summary
    """

    health = calculate_health_score()

    trend = calculate_trend()

    drift = calculate_drift()

    score = health["score"]

    return {

        # =========================
        # Health
        # =========================

        "health_score": score,

        "status": get_health_status(score),

        "advice": get_health_advice(score),

        "recent7": health["recent7"]["accuracy"],

        "recent30": health["recent30"]["accuracy"],

        "overall": health["overall"]["accuracy"],

        "validated": health["overall"]["total"],

        # =========================
        # Trend
        # =========================

        "trend": trend,

        # =========================
        # Drift
        # =========================

        "drift": drift,

        # =========================
        # AI Report
        # =========================

        "summary": generate_ai_summary(),
    }


def get_model_health():
    """
    與舊版相容

    舊程式不用修改即可運作

    之後 app.py 可慢慢改成：

        get_model_monitor()

    最後再移除此函式。
    """

    return get_model_monitor()