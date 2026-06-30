import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def load_valid_prediction_data(
    csv_path="prediction_log.csv"
):
    """
    讀取並清理已驗證的預測資料
    """

    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig"
    )

    df = df.dropna(
        subset=[
            "實際漲跌",
            "預測結果"
        ]
    )

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

    df = df[
        df["實際漲跌"].isin(
            [
                "上漲",
                "下跌"
            ]
        )
    ]

    df = df[
        df["預測結果"].isin(
            [
                "上漲",
                "下跌"
            ]
        )
    ]

    return df


def evaluate_model(
    csv_path="prediction_log.csv"
):
    """
    模型整體評估
    """

    df = load_valid_prediction_data(
        csv_path
    )

    if df.empty:
        return {
            "total": 0,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "confusion_matrix": [
                [0, 0],
                [0, 0]
            ]
        }

    y_true = (
        df["實際漲跌"]
        .map({
            "上漲": 1,
            "下跌": 0
        })
    )

    y_pred = (
        df["預測結果"]
        .map({
            "上漲": 1,
            "下跌": 0
        })
    )

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
            1
        ]
    )

    return {
        "total": len(df),
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "confusion_matrix": matrix.tolist()
    }

def evaluate_confidence_bins(
    csv_path="prediction_log.csv"
):
    """
    評估不同信心區間的實際準確率
    """

    df = load_valid_prediction_data(
        csv_path
    )

    if df.empty:
        return []

    df["信心值"] = pd.to_numeric(
        df["信心值"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["信心值"]
    )

    bins = [
        0,
        60,
        70,
        80,
        100
    ]

    labels = [
        "50~60%",
        "60~70%",
        "70~80%",
        "80%以上"
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
            group["是否預測正確"] == "正確"
        ).sum()

        accuracy = float(
            round(
                correct / total * 100,
                2
    )
)

        result.append({
            "range": label,
            "total": total,
            "correct": int(correct),
            "accuracy": accuracy
        })

    return result

def evaluate_stock_accuracy(
    csv_path="prediction_log.csv"
):
    """
    各股票模型準確率
    """

    df = load_valid_prediction_data(
        csv_path
    )

    if df.empty:
        return []

    result = []

    grouped = df.groupby("股票代號")

    for stock_id, group in grouped:

        total = len(group)

        if total < 5:
            continue

        correct = (
            group["是否預測正確"] == "正確"
        ).sum()

        accuracy = float(
            round(
                correct / total * 100,
                2
            )
        )

        stock_name = (
            group.iloc[0]["股票名稱"]
        )

        result.append({

            "stock_id": stock_id,

            "stock_name": stock_name,

            "total": total,

            "correct": int(correct),

            "accuracy": accuracy

        })

    result.sort(

        key=lambda x: x["accuracy"],

        reverse=True

    )

    return result


if __name__ == "__main__":

    stock_result = evaluate_stock_accuracy()

    print("\n========== 個股模型表現 ==========")

    for stock in stock_result:

        print(

            f"{stock['stock_name']}"

            f" ({stock['stock_id']})"

            f"  "

            f"{stock['accuracy']}%"

            f" ({stock['correct']}/{stock['total']})"

        )