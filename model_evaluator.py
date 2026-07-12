import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from database import SessionLocal
from models import Prediction


# ==========================
# 從 SQLite 讀取已驗證資料
# ==========================
def load_valid_prediction_data():
    """
    從 SQLite 讀取並清理已驗證的預測資料。

    只保留：
    - 實際漲跌為「上漲」或「下跌」
    - 預測結果為「上漲」或「下跌」
    """

    db = SessionLocal()

    try:
        predictions = (
            db.query(Prediction)
            .filter(
                Prediction.actual_change.in_(
                    ["上漲", "下跌"]
                ),
                Prediction.prediction_text.in_(
                    ["上漲", "下跌"]
                ),
            )
            .order_by(
                Prediction.predict_date.asc(),
                Prediction.id.asc(),
            )
            .all()
        )

        if not predictions:
            return pd.DataFrame()

        data = []

        for prediction in predictions:
            data.append({
                "預測日期": prediction.predict_date,
                "股票代號": prediction.stock_code,
                "股票名稱": prediction.stock_name,
                "預測結果": prediction.prediction_text,
                "信心值": prediction.confidence,
                "上漲機率": prediction.up_probability,
                "下跌機率": prediction.down_probability,
                "隔日預測參考價": prediction.predict_close,
                "預測區間下緣": prediction.lower_price,
                "預測區間上緣": prediction.upper_price,
                "實際收盤價": prediction.actual_close,
                "實際漲跌": prediction.actual_change,
                "是否預測正確": prediction.is_correct,
            })

        df = pd.DataFrame(data)

        df["實際漲跌"] = (
            df["實際漲跌"]
            .astype(str)
            .str.strip()
        )

        df["預測結果"] = (
            df["預測結果"]
            .astype(str)
            .str.strip()
        )

        return df

    except Exception as e:
        print(
            f"❌ SQLite 模型評估資料讀取失敗：{e}"
        )

        return pd.DataFrame()

    finally:
        db.close()


# ==========================
# 模型整體評估
# ==========================
def evaluate_model():
    """
    計算模型整體評估指標：

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
    """

    df = load_valid_prediction_data()

    if df.empty:
        return {
            "total": 0,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "confusion_matrix": [
                [0, 0],
                [0, 0],
            ],
        }

    y_true = df["實際漲跌"].map({
        "上漲": 1,
        "下跌": 0,
    })

    y_pred = df["預測結果"].map({
        "上漲": 1,
        "下跌": 0,
    })

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ]
    )

    return {
        "total": len(df),
        "accuracy": round(
            accuracy * 100,
            2
        ),
        "precision": round(
            precision * 100,
            2
        ),
        "recall": round(
            recall * 100,
            2
        ),
        "f1": round(
            f1 * 100,
            2
        ),
        "confusion_matrix": matrix.tolist(),
    }


# ==========================
# 信心區間分析
# ==========================
def evaluate_confidence_bins():
    """
    評估不同信心區間的實際準確率。
    """

    df = load_valid_prediction_data()

    if df.empty:
        return []

    df["信心值"] = pd.to_numeric(
        df["信心值"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["信心值"]
    )

    if df.empty:
        return []

    bins = [
        0,
        60,
        70,
        80,
        100,
    ]

    labels = [
        "50~60%",
        "60~70%",
        "70~80%",
        "80%以上",
    ]

    df["信心區間"] = pd.cut(
        df["信心值"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    result = []

    for label in labels:
        group = df[
            df["信心區間"] == label
        ]

        if group.empty:
            continue

        total = len(group)

        correct = (
            group["是否預測正確"]
            == "正確"
        ).sum()

        accuracy = round(
            correct / total * 100,
            2
        )

        result.append({
            "range": label,
            "total": total,
            "correct": int(correct),
            "accuracy": float(accuracy),
        })

    return result


# ==========================
# 各股票模型準確率
# ==========================
def evaluate_stock_accuracy():
    """
    計算各股票模型準確率。

    只顯示至少有 5 筆有效驗證資料的股票。
    """

    df = load_valid_prediction_data()

    if df.empty:
        return []

    result = []

    grouped = df.groupby(
        "股票代號"
    )

    for stock_id, group in grouped:
        total = len(group)

        if total < 5:
            continue

        correct = (
            group["是否預測正確"]
            == "正確"
        ).sum()

        accuracy = round(
            correct / total * 100,
            2
        )

        stock_name = (
            group.iloc[0]["股票名稱"]
        )

        result.append({
            "stock_id": str(stock_id),
            "stock_name": stock_name,
            "total": int(total),
            "correct": int(correct),
            "accuracy": float(accuracy),
        })

    result.sort(
        key=lambda item: (
            item["accuracy"],
            item["total"],
        ),
        reverse=True
    )

    return result


# ==========================
# 手動測試
# ==========================
if __name__ == "__main__":
    metrics = evaluate_model()
    confidence_result = evaluate_confidence_bins()
    stock_result = evaluate_stock_accuracy()

    print("\n========== 模型整體評估 ==========")
    print(metrics)

    print("\n========== 信心區間分析 ==========")

    for item in confidence_result:
        print(
            f"{item['range']}："
            f"{item['accuracy']}% "
            f"({item['correct']}/{item['total']})"
        )

    print("\n========== 個股模型表現 ==========")

    for stock in stock_result:
        print(
            f"{stock['stock_name']} "
            f"({stock['stock_id']}) "
            f"{stock['accuracy']}% "
            f"({stock['correct']}/{stock['total']})"
        )