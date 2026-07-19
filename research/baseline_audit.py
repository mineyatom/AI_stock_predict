"""
V10.0 Baseline Audit

用途：
1. 不修改正式 predictor.py。
2. 使用目前正式 XGBoost 特徵與參數。
3. 比較兩種資料區間：
   - Baseline A：沿用目前 2020 年至今資料，缺少的法人/大盤/美股資料已由 predictor.py 補 0。
   - Baseline B：只使用 2024-01-01 之後的共同資料區間。
4. 使用 Expanding Window 進行時間序列樣本外驗證。
5. 輸出 Accuracy、Precision、Recall、F1、Brier Score、Log Loss、混淆矩陣與信心區間勝率。

執行方式（專案根目錄）：
    D:\conda_envs\stock_ai\python.exe research\baseline_audit.py 2330

指定每次重新訓練間隔：
    D:\conda_envs\stock_ai\python.exe research\baseline_audit.py 2330 --retrain-every 20

輸出 CSV：
    research/results/baseline_2330_TW_predictions.csv
    research/results/baseline_2330_TW_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


# 讓 research/ 內的程式可以匯入專案根目錄的 predictor.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor import build_feature_data, resolve_stock_code  # noqa: E402


FEATURE_COLUMNS = [
    "Volume_MA5",
    "Return",
    "Return_1",
    "Return_2",
    "Return_3",
    "K",
    "D",
    "Foreign_Investor",
    "Investment_Trust",
    "Dealer_self",
    "Dealer_Hedging",
    "Market_Return",
    "Market_RSI",
    "Market_Volatility",
    "NVDA_Return",
    "SOX_Return",
    "QQQ_Return",
]


@dataclass(frozen=True)
class AuditConfig:
    minimum_train_size: int = 252
    retrain_every: int = 20
    common_start_date: str = "2024-01-01"


def create_model() -> XGBClassifier:
    """
    使用目前 predictor.py 的正式模型參數。

    額外指定 eval_metric，避免不同 XGBoost 版本出現提示；
    不改變 predictor.py 的正式模型。
    """
    return XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss",
    )


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    整理成可進行歷史驗證的資料。

    注意：
    最後一筆 Tomorrow_Close 為 NaN，不能拿來當歷史真實答案。
    """
    required_columns = [
        "date",
        "Close",
        "Tomorrow_Close",
        "Target",
        *FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "缺少必要欄位："
            + ", ".join(missing_columns)
        )

    model_df = df[required_columns].copy()

    model_df["date"] = pd.to_datetime(
        model_df["date"],
        errors="coerce",
    )

    for column in [
        "Close",
        "Tomorrow_Close",
        "Target",
        *FEATURE_COLUMNS,
    ]:
        model_df[column] = pd.to_numeric(
            model_df[column],
            errors="coerce",
        )

    model_df = model_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    model_df = model_df.dropna(
        subset=[
            "date",
            "Close",
            "Tomorrow_Close",
            "Target",
            *FEATURE_COLUMNS,
        ]
    )

    model_df = model_df.sort_values("date")
    model_df = model_df.drop_duplicates(
        subset=["date"],
        keep="last",
    )
    model_df = model_df.reset_index(drop=True)

    model_df["Target"] = model_df["Target"].astype(int)

    return model_df


def expanding_window_predictions(
    model_df: pd.DataFrame,
    *,
    dataset_name: str,
    minimum_train_size: int,
    retrain_every: int,
) -> pd.DataFrame:
    """
    Expanding Window 驗證。

    規則：
    - 前 minimum_train_size 筆只作為初始訓練資料。
    - 每 retrain_every 個交易日重新訓練一次。
    - 模型只能看到預測日期以前的資料。
    - 同一批次內的模型固定，不偷看後面的真實答案。
    """
    if retrain_every < 1:
        raise ValueError("retrain_every 必須至少為 1")

    if len(model_df) <= minimum_train_size:
        raise ValueError(
            f"{dataset_name} 樣本不足："
            f"目前 {len(model_df)} 筆，"
            f"至少需要 {minimum_train_size + 1} 筆。"
        )

    prediction_rows: list[dict[str, object]] = []

    start_index = minimum_train_size

    while start_index < len(model_df):
        end_index = min(
            start_index + retrain_every,
            len(model_df),
        )

        train_df = model_df.iloc[:start_index].copy()
        test_df = model_df.iloc[start_index:end_index].copy()

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["Target"]

        # 某些極端資料區間可能只有單一類別，無法建立二元分類模型。
        if y_train.nunique() < 2:
            raise ValueError(
                f"{dataset_name} 在 {test_df.iloc[0]['date'].date()} "
                "以前的訓練資料只有單一 Target 類別。"
            )

        model = create_model()
        model.fit(X_train, y_train)

        X_test = test_df[FEATURE_COLUMNS]

        up_probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (up_probabilities >= 0.5).astype(int)

        for row_position, (_, row) in enumerate(test_df.iterrows()):
            up_probability = float(up_probabilities[row_position])
            prediction = int(predictions[row_position])
            actual = int(row["Target"])

            confidence = (
                up_probability
                if prediction == 1
                else 1.0 - up_probability
            )

            prediction_rows.append(
                {
                    "dataset": dataset_name,
                    "date": row["date"].date().isoformat(),
                    "actual": actual,
                    "prediction": prediction,
                    "correct": int(prediction == actual),
                    "up_probability": up_probability,
                    "down_probability": 1.0 - up_probability,
                    "confidence": confidence,
                    "train_samples": len(train_df),
                    "train_start": (
                        train_df.iloc[0]["date"]
                        .date()
                        .isoformat()
                    ),
                    "train_end": (
                        train_df.iloc[-1]["date"]
                        .date()
                        .isoformat()
                    ),
                }
            )

        start_index = end_index

    return pd.DataFrame(prediction_rows)


def safe_log_loss(
    actual: pd.Series,
    probabilities: pd.Series,
) -> float:
    clipped_probabilities = np.clip(
        probabilities.astype(float),
        1e-6,
        1.0 - 1e-6,
    )

    return float(
        log_loss(
            actual,
            clipped_probabilities,
            labels=[0, 1],
        )
    )


def calculate_metrics(
    predictions_df: pd.DataFrame,
) -> dict[str, object]:
    actual = predictions_df["actual"].astype(int)
    prediction = predictions_df["prediction"].astype(int)
    probability = predictions_df["up_probability"].astype(float)

    matrix = confusion_matrix(
        actual,
        prediction,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "samples": len(predictions_df),
        "accuracy": accuracy_score(actual, prediction),
        "precision": precision_score(
            actual,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            actual,
            prediction,
            zero_division=0,
        ),
        "brier_score": brier_score_loss(
            actual,
            probability,
        ),
        "log_loss": safe_log_loss(
            actual,
            probability,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "actual_up_rate": actual.mean(),
        "predicted_up_rate": prediction.mean(),
        "average_confidence": predictions_df[
            "confidence"
        ].mean(),
    }


def build_confidence_bins(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    檢查模型信心值是否對應真實勝率。

    例如：
    70%～80% 信心的預測，實際正確率是否接近該範圍。
    """
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.000001]

    labels = [
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    result = predictions_df.copy()

    result["confidence_bin"] = pd.cut(
        result["confidence"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    grouped = (
        result.dropna(subset=["confidence_bin"])
        .groupby(
            "confidence_bin",
            observed=False,
        )
        .agg(
            samples=("correct", "size"),
            actual_accuracy=("correct", "mean"),
            average_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    grouped["calibration_gap"] = (
        grouped["average_confidence"]
        - grouped["actual_accuracy"]
    )

    return grouped


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_metrics(
    dataset_name: str,
    model_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    metrics: dict[str, object],
) -> None:
    print()
    print("=" * 68)
    print(f"V10.0 Baseline Audit｜{dataset_name}")
    print("=" * 68)
    print(
        "可用資料區間："
        f"{model_df.iloc[0]['date'].date()} "
        f"至 {model_df.iloc[-1]['date'].date()}"
    )
    print(f"原始有效樣本：{len(model_df)}")
    print(f"樣本外預測數：{metrics['samples']}")
    print("-" * 68)
    print(f"Accuracy       ：{format_percent(float(metrics['accuracy']))}")
    print(f"Precision      ：{format_percent(float(metrics['precision']))}")
    print(f"Recall         ：{format_percent(float(metrics['recall']))}")
    print(f"F1 Score       ：{format_percent(float(metrics['f1']))}")
    print(f"Brier Score    ：{float(metrics['brier_score']):.6f}")
    print(f"Log Loss       ：{float(metrics['log_loss']):.6f}")
    print(f"平均信心值     ：{format_percent(float(metrics['average_confidence']))}")
    print(f"實際上漲比例   ：{format_percent(float(metrics['actual_up_rate']))}")
    print(f"預測上漲比例   ：{format_percent(float(metrics['predicted_up_rate']))}")
    print("-" * 68)
    print("Confusion Matrix（列＝實際、欄＝預測）")
    print(
        np.array(
            [
                [
                    metrics["true_negative"],
                    metrics["false_positive"],
                ],
                [
                    metrics["false_negative"],
                    metrics["true_positive"],
                ],
            ]
        )
    )
    print("-" * 68)

    confidence_bins = build_confidence_bins(predictions_df)

    if confidence_bins.empty:
        print("沒有足夠資料建立信心區間統計。")
    else:
        printable_bins = confidence_bins.copy()

        for column in [
            "actual_accuracy",
            "average_confidence",
            "calibration_gap",
        ]:
            printable_bins[column] = printable_bins[column].map(
                lambda value: (
                    format_percent(float(value))
                    if pd.notna(value)
                    else "-"
                )
            )

        print("信心區間實際勝率")
        print(
            printable_bins.to_string(
                index=False,
            )
        )


def create_summary_row(
    stock_code: str,
    dataset_name: str,
    model_df: pd.DataFrame,
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "stock_code": stock_code,
        "dataset": dataset_name,
        "data_start": model_df.iloc[0]["date"].date().isoformat(),
        "data_end": model_df.iloc[-1]["date"].date().isoformat(),
        "available_samples": len(model_df),
        **metrics,
    }


def run_audit(
    stock_id: str,
    config: AuditConfig,
) -> None:
    stock_code = resolve_stock_code(stock_id)

    print("=" * 68)
    print("V10.0 Baseline Audit")
    print("=" * 68)
    print(f"股票代號：{stock_code}")
    print("正在建立正式模型使用的特徵資料……")

    raw_df = build_feature_data(stock_code)
    full_model_df = prepare_model_data(raw_df)

    baseline_a_df = full_model_df.copy()

    common_start = pd.Timestamp(config.common_start_date)

    baseline_b_df = full_model_df[
        full_model_df["date"] >= common_start
    ].copy()
    baseline_b_df = baseline_b_df.reset_index(drop=True)

    datasets: Iterable[tuple[str, pd.DataFrame]] = [
        (
            "Baseline A｜2020起，缺失外部特徵補0",
            baseline_a_df,
        ),
        (
            f"Baseline B｜{config.common_start_date}起共同資料區間",
            baseline_b_df,
        ),
    ]

    all_predictions: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for dataset_name, model_df in datasets:
        predictions_df = expanding_window_predictions(
            model_df,
            dataset_name=dataset_name,
            minimum_train_size=config.minimum_train_size,
            retrain_every=config.retrain_every,
        )

        metrics = calculate_metrics(predictions_df)

        print_metrics(
            dataset_name,
            model_df,
            predictions_df,
            metrics,
        )

        all_predictions.append(predictions_df)

        summary_rows.append(
            create_summary_row(
                stock_code,
                dataset_name,
                model_df,
                metrics,
            )
        )

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_stock_code = stock_code.replace(".", "_")

    predictions_output = (
        results_dir
        / f"baseline_{safe_stock_code}_predictions.csv"
    )

    summary_output = (
        results_dir
        / f"baseline_{safe_stock_code}_summary.csv"
    )

    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(summary_rows)

    combined_predictions.to_csv(
        predictions_output,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 68)
    print("兩組 Baseline 比較")
    print("=" * 68)

    display_columns = [
        "dataset",
        "samples",
        "accuracy",
        "f1",
        "brier_score",
        "log_loss",
        "average_confidence",
    ]

    comparison_df = summary_df[display_columns].copy()

    for column in [
        "accuracy",
        "f1",
        "average_confidence",
    ]:
        comparison_df[column] = comparison_df[column].map(
            lambda value: format_percent(float(value))
        )

    for column in [
        "brier_score",
        "log_loss",
    ]:
        comparison_df[column] = comparison_df[column].map(
            lambda value: f"{float(value):.6f}"
        )

    print(comparison_df.to_string(index=False))
    print()
    print(f"預測明細已輸出：{predictions_output}")
    print(f"摘要結果已輸出：{summary_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V10.0 XGBoost Baseline Audit",
    )

    parser.add_argument(
        "stock_id",
        help="股票代號，例如 2330 或 2330.TW",
    )

    parser.add_argument(
        "--minimum-train-size",
        type=int,
        default=252,
        help="初始訓練樣本數，預設 252",
    )

    parser.add_argument(
        "--retrain-every",
        type=int,
        default=20,
        help="每幾個交易日重新訓練，預設 20",
    )

    parser.add_argument(
        "--common-start-date",
        default="2024-01-01",
        help="Baseline B 起始日期，預設 2024-01-01",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = AuditConfig(
        minimum_train_size=args.minimum_train_size,
        retrain_every=args.retrain_every,
        common_start_date=args.common_start_date,
    )

    run_audit(
        args.stock_id,
        config,
    )


if __name__ == "__main__":
    main()
