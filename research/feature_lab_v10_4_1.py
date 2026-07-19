"""
V10.4 Feature Engineering Lab

目的：
1. 不修改正式 predictor.py。
2. 沿用正式 build_feature_data() 產生的既有資料與 Baseline Features。
3. 在相同時間區間、相同 Expanding Window 下，逐一比較新特徵。
4. 每次只加入一個候選特徵，避免無法判斷真正貢獻。
5. 評估 Accuracy、Precision、Recall、F1、Brier Score、Log Loss。
6. 輸出 Feature Ranking 與逐日樣本外預測。

候選特徵：
- ATR
- ADX
- OBV
- Bollinger Width
- MFI
- CCI
- ROC

執行方式：
    D:\conda_envs\stock_ai\python.exe research\feature_lab.py 2330

指定資料起始日：
    D:\conda_envs\stock_ai\python.exe research\feature_lab.py 2330 --start-date 2020-01-01

加快測試：
    D:\conda_envs\stock_ai\python.exe research\feature_lab.py 2330 --step 5

只測指定特徵：
    D:\conda_envs\stock_ai\python.exe research\feature_lab.py 2330 --features ATR OBV ADX

輸出：
    research/results/feature_lab_2330_TW_summary.csv
    research/results/feature_lab_2330_TW_predictions.csv
    research/results/feature_lab_2330_TW_feature_correlations.csv

重要：
- 預設逐日 Walk-Forward，執行時間較長。
- --step 5 只適合快速篩選，不等同正式逐日結果。
"""

from __future__ import annotations

SCRIPT_VERSION = "V10.4.1-common-date-fix"

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor import build_feature_data, resolve_stock_code  # noqa: E402


BASELINE_FEATURES = [
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


MODEL_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": 42,
    "eval_metric": "logloss",
    "n_jobs": -1,
}


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["Close"].shift(1)

    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["ATR_14"] = true_range(result).rolling(14).mean()
    result["ATR_Ratio_14"] = safe_divide(
        result["ATR_14"],
        result["Close"],
    )
    return result


def add_adx(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    high_diff = result["High"].diff()
    low_diff = -result["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (high_diff > low_diff) & (high_diff > 0),
            high_diff,
            0.0,
        ),
        index=result.index,
    )

    minus_dm = pd.Series(
        np.where(
            (low_diff > high_diff) & (low_diff > 0),
            low_diff,
            0.0,
        ),
        index=result.index,
    )

    atr = true_range(result).rolling(14).mean()

    plus_di = 100 * safe_divide(
        plus_dm.rolling(14).mean(),
        atr,
    )

    minus_di = 100 * safe_divide(
        minus_dm.rolling(14).mean(),
        atr,
    )

    dx = 100 * safe_divide(
        (plus_di - minus_di).abs(),
        plus_di + minus_di,
    )

    result["ADX_14"] = dx.rolling(14).mean()
    result["PLUS_DI_14"] = plus_di
    result["MINUS_DI_14"] = minus_di

    return result


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    direction = np.sign(result["Close"].diff()).fillna(0)
    raw_obv = (direction * result["Volume"]).cumsum()

    result["OBV"] = raw_obv
    result["OBV_Return_5"] = raw_obv.pct_change(5)

    return result


def add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    middle = result["Close"].rolling(20).mean()
    standard_deviation = result["Close"].rolling(20).std()

    upper = middle + 2 * standard_deviation
    lower = middle - 2 * standard_deviation

    result["BB_Width_20"] = safe_divide(
        upper - lower,
        middle,
    )

    result["BB_Position_20"] = safe_divide(
        result["Close"] - lower,
        upper - lower,
    )

    return result


def add_mfi(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    typical_price = (
        result["High"]
        + result["Low"]
        + result["Close"]
    ) / 3

    money_flow = typical_price * result["Volume"]
    direction = typical_price.diff()

    positive_flow = money_flow.where(direction > 0, 0.0)
    negative_flow = money_flow.where(direction < 0, 0.0).abs()

    positive_sum = positive_flow.rolling(14).sum()
    negative_sum = negative_flow.rolling(14).sum()

    money_ratio = safe_divide(
        positive_sum,
        negative_sum,
    )

    result["MFI_14"] = 100 - (
        100 / (1 + money_ratio)
    )

    return result


def add_cci(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    typical_price = (
        result["High"]
        + result["Low"]
        + result["Close"]
    ) / 3

    moving_average = typical_price.rolling(20).mean()

    mean_deviation = typical_price.rolling(20).apply(
        lambda values: np.mean(
            np.abs(values - np.mean(values))
        ),
        raw=True,
    )

    result["CCI_20"] = safe_divide(
        typical_price - moving_average,
        0.015 * mean_deviation,
    )

    return result


def add_roc(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["ROC_10"] = result["Close"].pct_change(10)
    result["ROC_20"] = result["Close"].pct_change(20)

    return result


FEATURE_BUILDERS: dict[
    str,
    tuple[Callable[[pd.DataFrame], pd.DataFrame], list[str]],
] = {
    "ATR": (
        add_atr,
        ["ATR_14", "ATR_Ratio_14"],
    ),
    "ADX": (
        add_adx,
        ["ADX_14", "PLUS_DI_14", "MINUS_DI_14"],
    ),
    "OBV": (
        add_obv,
        ["OBV", "OBV_Return_5"],
    ),
    "BOLLINGER": (
        add_bollinger,
        ["BB_Width_20", "BB_Position_20"],
    ),
    "MFI": (
        add_mfi,
        ["MFI_14"],
    ),
    "CCI": (
        add_cci,
        ["CCI_20"],
    ),
    "ROC": (
        add_roc,
        ["ROC_10", "ROC_20"],
    ),
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    rename_map = {}

    for column in result.columns:
        normalized = str(column).strip()

        if normalized.lower() == "date":
            rename_map[column] = "date"
        elif normalized.lower() == "open":
            rename_map[column] = "Open"
        elif normalized.lower() == "high":
            rename_map[column] = "High"
        elif normalized.lower() == "low":
            rename_map[column] = "Low"
        elif normalized.lower() == "close":
            rename_map[column] = "Close"
        elif normalized.lower() == "volume":
            rename_map[column] = "Volume"

    return result.rename(columns=rename_map)


def prepare_base_dataframe(
    raw_df: pd.DataFrame,
    start_date: str | None,
) -> pd.DataFrame:
    df = normalize_columns(raw_df)

    required_price_columns = [
        "date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Target",
        *BASELINE_FEATURES,
    ]

    missing = [
        column
        for column in required_price_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "build_feature_data() 缺少必要欄位："
            + ", ".join(missing)
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Target",
        *BASELINE_FEATURES,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values("date")
    df = df.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    if start_date:
        df = df[
            df["date"] >= pd.Timestamp(start_date)
        ]

    df = df.reset_index(drop=True)

    return df


def create_model() -> XGBClassifier:
    return XGBClassifier(**MODEL_PARAMS)


def calculate_metrics(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    probabilities = np.clip(
        probabilities,
        1e-6,
        1.0 - 1e-6,
    )

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "samples": len(actual),
        "accuracy": accuracy_score(actual, predictions),
        "precision": precision_score(
            actual,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            actual,
            predictions,
            zero_division=0,
        ),
        "brier_score": brier_score_loss(
            actual,
            probabilities,
        ),
        "log_loss": log_loss(
            actual,
            probabilities,
            labels=[0, 1],
        ),
        "average_confidence": np.maximum(
            probabilities,
            1 - probabilities,
        ).mean(),
        "predicted_up_rate": predictions.mean(),
        "actual_up_rate": actual.mean(),
    }


def walk_forward_predict(
    df: pd.DataFrame,
    feature_columns: list[str],
    minimum_train_samples: int,
    step: int,
    experiment_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    required_columns = [
        "date",
        "Target",
        *feature_columns,
    ]

    model_df = df[required_columns].copy()
    model_df = model_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )
    model_df = model_df.dropna(
        subset=required_columns,
    )
    model_df["Target"] = model_df["Target"].astype(int)
    model_df = model_df.reset_index(drop=True)

    if len(model_df) <= minimum_train_samples:
        raise ValueError(
            f"{experiment_name} 可用資料只有 {len(model_df)} 筆，"
            f"不足 minimum_train_samples={minimum_train_samples}。"
        )

    rows = []

    total_predictions = (
        len(model_df) - minimum_train_samples
    )

    print(
        f"{experiment_name}: "
        f"{len(feature_columns)} features，"
        f"預計 {((total_predictions - 1) // step) + 1} 次訓練"
    )

    for test_index in range(
        minimum_train_samples,
        len(model_df),
        step,
    ):
        train_df = model_df.iloc[:test_index]
        test_df = model_df.iloc[[test_index]]

        X_train = train_df[feature_columns]
        y_train = train_df["Target"]

        if y_train.nunique() < 2:
            continue

        model = create_model()
        model.fit(X_train, y_train)

        probability = float(
            model.predict_proba(
                test_df[feature_columns]
            )[0, 1]
        )

        actual = int(test_df.iloc[0]["Target"])
        prediction = int(probability >= 0.5)

        rows.append(
            {
                "experiment": experiment_name,
                "date": test_df.iloc[0]["date"],
                "actual": actual,
                "prediction": prediction,
                "up_probability": probability,
                "confidence": max(
                    probability,
                    1 - probability,
                ),
                "correct": int(prediction == actual),
            }
        )

    predictions_df = pd.DataFrame(rows)

    if predictions_df.empty:
        raise ValueError(
            f"{experiment_name} 沒有產生任何預測。"
        )

    metrics = calculate_metrics(
        predictions_df["actual"].to_numpy(),
        predictions_df["up_probability"].to_numpy(),
    )

    return metrics, predictions_df


def align_to_common_dates(
    prediction_frames: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    common_dates = None

    for frame in prediction_frames:
        dates = set(
            pd.to_datetime(
                frame["date"],
                errors="coerce",
            ).dt.normalize().dropna().tolist()
        )

        common_dates = (
            dates
            if common_dates is None
            else common_dates & dates
        )

    if not common_dates:
        diagnostics = []

        for frame in prediction_frames:
            normalized_dates = pd.to_datetime(
                frame["date"],
                errors="coerce",
            ).dt.normalize().dropna()

            experiment = (
                str(frame.iloc[0]["experiment"])
                if not frame.empty
                else "UNKNOWN"
            )

            diagnostics.append(
                f"{experiment}: "
                f"samples={len(frame)}, "
                f"start={normalized_dates.min() if not normalized_dates.empty else 'NA'}, "
                f"end={normalized_dates.max() if not normalized_dates.empty else 'NA'}"
            )

        raise ValueError(
            "不同實驗之間沒有共同預測日期。\n"
            + "\n".join(diagnostics)
        )

    common_date_series = pd.Series(
        sorted(common_dates),
        name="date",
    )

    aligned = []

    for frame in prediction_frames:
        current = frame.copy()
        current["date"] = pd.to_datetime(
            current["date"],
            errors="coerce",
        ).dt.normalize()

        current = current[
            current["date"].isin(common_date_series)
        ].sort_values("date")

        aligned.append(current.reset_index(drop=True))

    return aligned


def build_summary(
    aligned_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for frame in aligned_frames:
        experiment_name = frame.iloc[0]["experiment"]

        metrics = calculate_metrics(
            frame["actual"].to_numpy(),
            frame["up_probability"].to_numpy(),
        )

        rows.append(
            {
                "experiment": experiment_name,
                **metrics,
            }
        )

    summary_df = pd.DataFrame(rows)

    baseline = summary_df[
        summary_df["experiment"] == "Baseline"
    ].iloc[0]

    summary_df["accuracy_delta"] = (
        summary_df["accuracy"]
        - baseline["accuracy"]
    )

    summary_df["f1_delta"] = (
        summary_df["f1"]
        - baseline["f1"]
    )

    summary_df["brier_delta"] = (
        summary_df["brier_score"]
        - baseline["brier_score"]
    )

    summary_df["log_loss_delta"] = (
        summary_df["log_loss"]
        - baseline["log_loss"]
    )

    def recommendation(row: pd.Series) -> str:
        if row["experiment"] == "Baseline":
            return "BASELINE"

        score = 0

        if row["accuracy_delta"] > 0:
            score += 1

        if row["f1_delta"] > 0:
            score += 1

        if row["brier_delta"] < 0:
            score += 1

        if row["log_loss_delta"] < 0:
            score += 1

        if (
            row["accuracy_delta"] >= 0.005
            and row["brier_delta"] <= 0
        ):
            score += 1

        if score >= 4:
            return "KEEP"

        if score >= 2:
            return "REVIEW"

        return "REMOVE"

    summary_df["recommendation"] = summary_df.apply(
        recommendation,
        axis=1,
    )

    summary_df = summary_df.sort_values(
        by=[
            "recommendation",
            "accuracy_delta",
            "f1_delta",
            "brier_delta",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return summary_df


def feature_correlation_table(
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []

    for experiment, columns in feature_groups.items():
        for column in columns:
            valid = df[
                [column, "Target"]
            ].replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            correlation = (
                valid[column].corr(valid["Target"])
                if len(valid) >= 2
                else np.nan
            )

            rows.append(
                {
                    "experiment": experiment,
                    "feature": column,
                    "samples": len(valid),
                    "target_correlation": correlation,
                    "missing_rate": df[column].isna().mean(),
                }
            )

    return pd.DataFrame(rows)


def print_summary(summary_df: pd.DataFrame) -> None:
    printable = summary_df.copy()

    percent_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "average_confidence",
        "predicted_up_rate",
        "actual_up_rate",
        "accuracy_delta",
        "f1_delta",
    ]

    for column in percent_columns:
        printable[column] = printable[column].map(
            lambda value: f"{float(value) * 100:.2f}%"
        )

    decimal_columns = [
        "brier_score",
        "log_loss",
        "brier_delta",
        "log_loss_delta",
    ]

    for column in decimal_columns:
        printable[column] = printable[column].map(
            lambda value: f"{float(value):.6f}"
        )

    print()
    print("=" * 110)
    print("V10.4 Feature Lab Summary")
    print("=" * 110)

    display_columns = [
        "experiment",
        "samples",
        "accuracy",
        "f1",
        "brier_score",
        "log_loss",
        "accuracy_delta",
        "f1_delta",
        "brier_delta",
        "log_loss_delta",
        "recommendation",
    ]

    print(
        printable[display_columns].to_string(
            index=False
        )
    )


def run_feature_lab(
    stock_id: str,
    start_date: str | None,
    selected_features: list[str],
    minimum_train_samples: int,
    step: int,
) -> None:
    stock_code = resolve_stock_code(stock_id)

    print("=" * 82)
    print(f"V10.4 Feature Engineering Lab｜{SCRIPT_VERSION}")
    print("=" * 82)
    print(f"股票代號：{stock_code}")
    print(f"候選特徵：{', '.join(selected_features)}")
    print("正在建立正式 Feature Data……")

    raw_df = build_feature_data(stock_code)
    base_df = prepare_base_dataframe(
        raw_df,
        start_date,
    )

    experiment_df = base_df.copy()

    feature_groups: dict[str, list[str]] = {}

    for feature_name in selected_features:
        builder, new_columns = FEATURE_BUILDERS[
            feature_name
        ]

        experiment_df = builder(experiment_df)
        feature_groups[feature_name] = new_columns

    # 先鎖定所有實驗共同可用的資料列。
    # 各技術指標的 rolling 前置期不同；若各自 dropna 後再套用 --step，
    # 會從不同起點抽樣，導致預測日期完全錯開。
    all_experiment_features = list(BASELINE_FEATURES)

    for columns in feature_groups.values():
        all_experiment_features.extend(columns)

    common_required_columns = [
        "date",
        "Target",
        *all_experiment_features,
    ]

    common_experiment_df = experiment_df[
        common_required_columns
    ].copy()

    common_experiment_df = common_experiment_df.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=common_required_columns,
    ).sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    if len(common_experiment_df) <= minimum_train_samples:
        raise ValueError(
            "所有實驗共同有效資料不足："
            f"{len(common_experiment_df)} 筆；"
            f"minimum_train_samples={minimum_train_samples}。"
        )

    print(
        "共同實驗資料："
        f"{common_experiment_df.iloc[0]['date'].date()} 至 "
        f"{common_experiment_df.iloc[-1]['date'].date()}，"
        f"共 {len(common_experiment_df)} 筆"
    )

    raw_prediction_frames = []

    baseline_metrics, baseline_predictions = (
        walk_forward_predict(
            common_experiment_df,
            BASELINE_FEATURES,
            minimum_train_samples,
            step,
            "Baseline",
        )
    )

    raw_prediction_frames.append(
        baseline_predictions
    )

    print(
        "Baseline 完成："
        f"Accuracy={baseline_metrics['accuracy']:.2%}，"
        f"F1={baseline_metrics['f1']:.2%}"
    )

    for feature_name in selected_features:
        new_columns = feature_groups[feature_name]
        experiment_features = (
            BASELINE_FEATURES + new_columns
        )

        metrics, predictions = walk_forward_predict(
            common_experiment_df,
            experiment_features,
            minimum_train_samples,
            step,
            feature_name,
        )

        raw_prediction_frames.append(predictions)

        print(
            f"{feature_name} 完成："
            f"Accuracy={metrics['accuracy']:.2%}，"
            f"F1={metrics['f1']:.2%}，"
            f"Brier={metrics['brier_score']:.6f}"
        )

    aligned_frames = align_to_common_dates(
        raw_prediction_frames
    )

    summary_df = build_summary(aligned_frames)

    predictions_df = pd.concat(
        aligned_frames,
        ignore_index=True,
    )

    correlations_df = feature_correlation_table(
        experiment_df,
        feature_groups,
    )

    results_dir = (
        Path(__file__).resolve().parent
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_stock_code = stock_code.replace(".", "_")

    summary_path = (
        results_dir
        / f"feature_lab_{safe_stock_code}_summary.csv"
    )

    predictions_path = (
        results_dir
        / f"feature_lab_{safe_stock_code}_predictions.csv"
    )

    correlations_path = (
        results_dir
        / f"feature_lab_{safe_stock_code}_feature_correlations.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    correlations_df.to_csv(
        correlations_path,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(summary_df)

    print()
    print("=" * 82)
    print("輸出完成")
    print("=" * 82)
    print(f"Feature Ranking：{summary_path}")
    print(f"逐日預測明細：{predictions_path}")
    print(f"Feature Correlation：{correlations_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V10.4 Feature Engineering "
            "Walk-Forward Lab"
        )
    )

    parser.add_argument(
        "stock_id",
        help="股票代號，例如 2330 或 2330.TW",
    )

    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="資料起始日，預設 2020-01-01",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=list(FEATURE_BUILDERS.keys()),
        help=(
            "指定候選特徵，例如 "
            "--features ATR OBV ADX"
        ),
    )

    parser.add_argument(
        "--minimum-train-samples",
        type=int,
        default=252,
        help="第一次預測前最少訓練樣本，預設 252",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help=(
            "Walk-Forward 間隔，預設 1；"
            "快速篩選可設 5"
        ),
    )

    args = parser.parse_args()

    normalized_features = [
        feature.upper()
        for feature in args.features
    ]

    unknown_features = [
        feature
        for feature in normalized_features
        if feature not in FEATURE_BUILDERS
    ]

    if unknown_features:
        parser.error(
            "不支援的 Feature："
            + ", ".join(unknown_features)
            + "；可用："
            + ", ".join(FEATURE_BUILDERS.keys())
        )

    if args.minimum_train_samples < 100:
        parser.error(
            "--minimum-train-samples 建議至少 100"
        )

    if args.step < 1:
        parser.error("--step 必須至少為 1")

    args.features = normalized_features

    return args


def main() -> None:
    args = parse_args()

    run_feature_lab(
        stock_id=args.stock_id,
        start_date=args.start_date,
        selected_features=args.features,
        minimum_train_samples=args.minimum_train_samples,
        step=args.step,
    )


if __name__ == "__main__":
    main()
