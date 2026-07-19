"""
V10.2 Probability Calibration Experiment

目的：
1. 使用 V10.0 產生的歷史樣本外預測，不修改正式 predictor.py。
2. 依照時間順序切分校正資料與測試資料，避免用測試答案校正自己。
3. 比較：
   - Raw：原始 XGBoost predict_proba
   - Platt Scaling
   - Isotonic Regression
4. 評估 Accuracy、Brier Score、Log Loss、ECE、MCE。
5. 每個 dataset 分開校正，避免 Baseline A/B 混在一起。

執行方式：
    D:\conda_envs\stock_ai\python.exe research\probability_calibration.py ^
        research\results\baseline_2330_TW_predictions.csv

指定前 60% 作校正、後 40% 作測試：
    D:\conda_envs\stock_ai\python.exe research\probability_calibration.py ^
        research\results\baseline_2330_TW_predictions.csv ^
        --calibration-ratio 0.60

輸出：
    *_calibration_predictions.csv
    *_calibration_summary.csv
    *_calibration_bins.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)


CONFIDENCE_BINS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.80,
    0.90,
    1.000001,
]


def clip_probability(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(
        np.asarray(values, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )


def probability_to_logit(values: np.ndarray | pd.Series) -> np.ndarray:
    probabilities = clip_probability(values)

    return np.log(
        probabilities / (1.0 - probabilities)
    ).reshape(-1, 1)


def calculate_confidence(
    probabilities: np.ndarray | pd.Series,
) -> np.ndarray:
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    return np.maximum(
        probabilities,
        1.0 - probabilities,
    )


def calculate_prediction(
    probabilities: np.ndarray | pd.Series,
) -> np.ndarray:
    return (
        np.asarray(probabilities, dtype=float) >= 0.5
    ).astype(int)


def calibration_metrics(
    actual: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=int)
    probability_array = clip_probability(probabilities)

    predictions = calculate_prediction(probability_array)
    confidence = calculate_confidence(probability_array)
    correct = (predictions == actual_array).astype(int)

    rows: list[dict[str, float]] = []

    for lower, upper in zip(
        CONFIDENCE_BINS[:-1],
        CONFIDENCE_BINS[1:],
    ):
        mask = (
            (confidence >= lower)
            & (confidence < upper)
        )

        if not mask.any():
            continue

        rows.append(
            {
                "samples": int(mask.sum()),
                "accuracy": float(correct[mask].mean()),
                "average_confidence": float(
                    confidence[mask].mean()
                ),
            }
        )

    total_samples = len(actual_array)

    ece = sum(
        row["samples"]
        / total_samples
        * abs(
            row["average_confidence"]
            - row["accuracy"]
        )
        for row in rows
    )

    mce = max(
        (
            abs(
                row["average_confidence"]
                - row["accuracy"]
            )
            for row in rows
        ),
        default=float("nan"),
    )

    return {
        "samples": int(total_samples),
        "accuracy": float(
            accuracy_score(actual_array, predictions)
        ),
        "average_confidence": float(confidence.mean()),
        "brier_score": float(
            brier_score_loss(
                actual_array,
                probability_array,
            )
        ),
        "log_loss": float(
            log_loss(
                actual_array,
                probability_array,
                labels=[0, 1],
            )
        ),
        "ece": float(ece),
        "mce": float(mce),
    }


def confidence_bin_table(
    dataset_name: str,
    method: str,
    actual: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
) -> pd.DataFrame:
    actual_array = np.asarray(actual, dtype=int)
    probability_array = clip_probability(probabilities)

    prediction = calculate_prediction(probability_array)
    confidence = calculate_confidence(probability_array)
    correct = (prediction == actual_array).astype(int)

    rows: list[dict[str, object]] = []

    for lower, upper in zip(
        CONFIDENCE_BINS[:-1],
        CONFIDENCE_BINS[1:],
    ):
        mask = (
            (confidence >= lower)
            & (confidence < upper)
        )

        label = (
            f"{int(lower * 100)}-"
            f"{int(min(upper, 1.0) * 100)}%"
        )

        if not mask.any():
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "confidence_bin": label,
                    "samples": 0,
                    "actual_accuracy": np.nan,
                    "average_confidence": np.nan,
                    "calibration_gap": np.nan,
                }
            )
            continue

        actual_accuracy = float(correct[mask].mean())
        average_confidence = float(confidence[mask].mean())

        rows.append(
            {
                "dataset": dataset_name,
                "method": method,
                "confidence_bin": label,
                "samples": int(mask.sum()),
                "actual_accuracy": actual_accuracy,
                "average_confidence": average_confidence,
                "calibration_gap": (
                    average_confidence
                    - actual_accuracy
                ),
            }
        )

    return pd.DataFrame(rows)


def fit_platt_scaler(
    probabilities: pd.Series,
    actual: pd.Series,
) -> LogisticRegression:
    """
    Platt Scaling：
    對原始機率的 logit 再做 Logistic Regression。
    """
    model = LogisticRegression(
        random_state=42,
        solver="lbfgs",
    )

    model.fit(
        probability_to_logit(probabilities),
        actual.astype(int),
    )

    return model


def apply_platt_scaler(
    model: LogisticRegression,
    probabilities: pd.Series,
) -> np.ndarray:
    return model.predict_proba(
        probability_to_logit(probabilities)
    )[:, 1]


def fit_isotonic_scaler(
    probabilities: pd.Series,
    actual: pd.Series,
) -> IsotonicRegression:
    model = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    )

    model.fit(
        probabilities.astype(float),
        actual.astype(int),
    )

    return model


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "dataset",
        "date",
        "actual",
        "up_probability",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "CSV 缺少必要欄位："
            + ", ".join(sorted(missing_columns))
        )

    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    result["actual"] = pd.to_numeric(
        result["actual"],
        errors="coerce",
    )

    result["up_probability"] = pd.to_numeric(
        result["up_probability"],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "dataset",
            "date",
            "actual",
            "up_probability",
        ]
    )

    result = result[
        result["actual"].isin([0, 1])
    ].copy()

    result["actual"] = result["actual"].astype(int)
    result["up_probability"] = clip_probability(
        result["up_probability"]
    )

    result = result.sort_values(
        ["dataset", "date"]
    ).reset_index(drop=True)

    return result


def split_calibration_test(
    dataset_df: pd.DataFrame,
    calibration_ratio: float,
    minimum_calibration_samples: int,
    minimum_test_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_samples = len(dataset_df)

    split_index = int(
        total_samples * calibration_ratio
    )

    split_index = max(
        split_index,
        minimum_calibration_samples,
    )

    split_index = min(
        split_index,
        total_samples - minimum_test_samples,
    )

    if split_index < minimum_calibration_samples:
        raise ValueError(
            "校正樣本不足。"
        )

    if total_samples - split_index < minimum_test_samples:
        raise ValueError(
            "測試樣本不足。"
        )

    calibration_df = (
        dataset_df.iloc[:split_index]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        dataset_df.iloc[split_index:]
        .copy()
        .reset_index(drop=True)
    )

    if calibration_df["actual"].nunique() < 2:
        raise ValueError(
            "校正資料只有單一類別，無法校正機率。"
        )

    return calibration_df, test_df


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_comparison(
    dataset_name: str,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    print()
    print("=" * 78)
    print(f"V10.2 Probability Calibration｜{dataset_name}")
    print("=" * 78)
    print(
        "校正資料："
        f"{calibration_df.iloc[0]['date'].date()} "
        f"至 {calibration_df.iloc[-1]['date'].date()}，"
        f"{len(calibration_df)} 筆"
    )
    print(
        "測試資料："
        f"{test_df.iloc[0]['date'].date()} "
        f"至 {test_df.iloc[-1]['date'].date()}，"
        f"{len(test_df)} 筆"
    )
    print("-" * 78)

    printable = summary_df[
        [
            "method",
            "samples",
            "accuracy",
            "average_confidence",
            "brier_score",
            "log_loss",
            "ece",
            "mce",
        ]
    ].copy()

    for column in [
        "accuracy",
        "average_confidence",
        "ece",
        "mce",
    ]:
        printable[column] = printable[column].map(
            lambda value: format_percent(float(value))
        )

    for column in [
        "brier_score",
        "log_loss",
    ]:
        printable[column] = printable[column].map(
            lambda value: f"{float(value):.6f}"
        )

    print(printable.to_string(index=False))

    raw_row = summary_df[
        summary_df["method"] == "Raw"
    ].iloc[0]

    print("-" * 78)
    print("相對 Raw 改善")

    for method in ["Platt", "Isotonic"]:
        method_row = summary_df[
            summary_df["method"] == method
        ].iloc[0]

        print(
            f"{method:8s}｜"
            f"Brier 改善 "
            f"{raw_row['brier_score'] - method_row['brier_score']:+.6f}｜"
            f"Log Loss 改善 "
            f"{raw_row['log_loss'] - method_row['log_loss']:+.6f}｜"
            f"ECE 改善 "
            f"{raw_row['ece'] - method_row['ece']:+.2%}"
        )


def run_experiment(
    input_path: Path,
    calibration_ratio: float,
    minimum_calibration_samples: int,
    minimum_test_samples: int,
) -> None:
    raw_df = pd.read_csv(input_path)
    df = validate_input(raw_df)

    all_prediction_frames: list[pd.DataFrame] = []
    all_summary_rows: list[dict[str, object]] = []
    all_bin_frames: list[pd.DataFrame] = []

    for dataset_name, dataset_df in df.groupby(
        "dataset",
        sort=False,
    ):
        dataset_df = dataset_df.sort_values(
            "date"
        ).reset_index(drop=True)

        calibration_df, test_df = split_calibration_test(
            dataset_df,
            calibration_ratio,
            minimum_calibration_samples,
            minimum_test_samples,
        )

        platt_model = fit_platt_scaler(
            calibration_df["up_probability"],
            calibration_df["actual"],
        )

        isotonic_model = fit_isotonic_scaler(
            calibration_df["up_probability"],
            calibration_df["actual"],
        )

        method_probabilities = {
            "Raw": clip_probability(
                test_df["up_probability"]
            ),
            "Platt": apply_platt_scaler(
                platt_model,
                test_df["up_probability"],
            ),
            "Isotonic": isotonic_model.predict(
                test_df["up_probability"].astype(float)
            ),
        }

        dataset_summary_rows: list[dict[str, object]] = []

        output_predictions = test_df[
            [
                "dataset",
                "date",
                "actual",
                "up_probability",
            ]
        ].copy()

        output_predictions = output_predictions.rename(
            columns={
                "up_probability": "raw_probability",
            }
        )

        for method, probabilities in method_probabilities.items():
            probabilities = clip_probability(probabilities)

            metrics = calibration_metrics(
                test_df["actual"],
                probabilities,
            )

            summary_row = {
                "dataset": dataset_name,
                "method": method,
                "calibration_start": (
                    calibration_df.iloc[0]["date"]
                    .date()
                    .isoformat()
                ),
                "calibration_end": (
                    calibration_df.iloc[-1]["date"]
                    .date()
                    .isoformat()
                ),
                "test_start": (
                    test_df.iloc[0]["date"]
                    .date()
                    .isoformat()
                ),
                "test_end": (
                    test_df.iloc[-1]["date"]
                    .date()
                    .isoformat()
                ),
                **metrics,
            }

            dataset_summary_rows.append(summary_row)
            all_summary_rows.append(summary_row)

            method_key = method.lower()

            output_predictions[
                f"{method_key}_probability"
            ] = probabilities

            output_predictions[
                f"{method_key}_prediction"
            ] = calculate_prediction(probabilities)

            output_predictions[
                f"{method_key}_confidence"
            ] = calculate_confidence(probabilities)

            all_bin_frames.append(
                confidence_bin_table(
                    dataset_name,
                    method,
                    test_df["actual"],
                    probabilities,
                )
            )

        dataset_summary_df = pd.DataFrame(
            dataset_summary_rows
        )

        print_comparison(
            dataset_name,
            calibration_df,
            test_df,
            dataset_summary_df,
        )

        all_prediction_frames.append(
            output_predictions
        )

    output_stem = input_path.stem

    predictions_output = input_path.with_name(
        f"{output_stem}_calibration_predictions.csv"
    )

    summary_output = input_path.with_name(
        f"{output_stem}_calibration_summary.csv"
    )

    bins_output = input_path.with_name(
        f"{output_stem}_calibration_bins.csv"
    )

    pd.concat(
        all_prediction_frames,
        ignore_index=True,
    ).to_csv(
        predictions_output,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        all_summary_rows
    ).to_csv(
        summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        all_bin_frames,
        ignore_index=True,
    ).to_csv(
        bins_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print("輸出完成")
    print("=" * 78)
    print(f"測試預測明細：{predictions_output}")
    print(f"校正比較摘要：{summary_output}")
    print(f"信心區間明細：{bins_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V10.2 Platt / Isotonic "
            "Probability Calibration Experiment"
        )
    )

    parser.add_argument(
        "csv",
        help=(
            "V10.0 產生的 "
            "baseline_*_predictions.csv"
        ),
    )

    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.60,
        help=(
            "依時間排序後，前段用於校正的比例；"
            "預設 0.60"
        ),
    )

    parser.add_argument(
        "--minimum-calibration-samples",
        type=int,
        default=100,
        help="最少校正樣本數，預設 100",
    )

    parser.add_argument(
        "--minimum-test-samples",
        type=int,
        default=50,
        help="最少測試樣本數，預設 50",
    )

    args = parser.parse_args()

    if not 0.20 <= args.calibration_ratio <= 0.80:
        parser.error(
            "--calibration-ratio 必須介於 0.20 到 0.80"
        )

    return args


def main() -> None:
    args = parse_args()

    run_experiment(
        Path(args.csv),
        args.calibration_ratio,
        args.minimum_calibration_samples,
        args.minimum_test_samples,
    )


if __name__ == "__main__":
    main()
