"""
V10.3 XGBoost Hyperparameter Search

目的：
1. 不修改正式 predictor.py。
2. 沿用目前正式 Feature。
3. 使用 TimeSeriesSplit + RandomizedSearchCV，避免一般 KFold 的時間洩漏。
4. 將最後 20% 歷史資料保留為 Final Holdout，不參與參數搜尋。
5. 比較：
   - Current Baseline Parameters
   - Best RandomizedSearch Parameters
6. 評估 Accuracy、Precision、Recall、F1、Brier Score、Log Loss。

執行方式：
    D:\conda_envs\stock_ai\python.exe research\hyperparameter_search.py 2330

增加搜尋次數：
    D:\conda_envs\stock_ai\python.exe research\hyperparameter_search.py 2330 --n-iter 80

指定資料起始日：
    D:\conda_envs\stock_ai\python.exe research\hyperparameter_search.py 2330 --start-date 2020-01-01

輸出：
    research/results/hyperparameter_2330_TW_summary.csv
    research/results/hyperparameter_2330_TW_best_params.json
    research/results/hyperparameter_2330_TW_search_results.csv
    research/results/hyperparameter_2330_TW_holdout_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier


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


BASELINE_PARAMS: dict[str, Any] = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": 42,
    "eval_metric": "logloss",
    "n_jobs": -1,
}


def create_baseline_model() -> XGBClassifier:
    return XGBClassifier(**BASELINE_PARAMS)


def create_search_model() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
        n_jobs=1,
    )


def prepare_data(
    df: pd.DataFrame,
    start_date: str | None,
) -> pd.DataFrame:
    required_columns = [
        "date",
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

    numeric_columns = [
        "Tomorrow_Close",
        "Target",
        *FEATURE_COLUMNS,
    ]

    for column in numeric_columns:
        model_df[column] = pd.to_numeric(
            model_df[column],
            errors="coerce",
        )

    model_df = model_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    model_df = model_df.dropna(
        subset=required_columns,
    )

    model_df = model_df.sort_values("date")
    model_df = model_df.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    if start_date:
        model_df = model_df[
            model_df["date"] >= pd.Timestamp(start_date)
        ]

    model_df = model_df.reset_index(drop=True)
    model_df["Target"] = model_df["Target"].astype(int)

    if model_df["Target"].nunique() < 2:
        raise ValueError("Target 只有單一類別，無法訓練分類模型。")

    return model_df


def split_train_holdout(
    model_df: pd.DataFrame,
    holdout_ratio: float,
    minimum_holdout_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_size = max(
        int(len(model_df) * holdout_ratio),
        minimum_holdout_samples,
    )

    if holdout_size >= len(model_df):
        raise ValueError("Holdout 太大，沒有剩餘訓練資料。")

    split_index = len(model_df) - holdout_size

    train_df = model_df.iloc[:split_index].copy()
    holdout_df = model_df.iloc[split_index:].copy()

    if len(train_df) < 300:
        raise ValueError(
            f"搜尋訓練資料只有 {len(train_df)} 筆，建議至少 300 筆。"
        )

    if train_df["Target"].nunique() < 2:
        raise ValueError("搜尋訓練資料只有單一 Target 類別。")

    return (
        train_df.reset_index(drop=True),
        holdout_df.reset_index(drop=True),
    )


def build_parameter_distributions() -> dict[str, Any]:
    """
    搜尋範圍刻意保持中等，避免超大型搜尋拖垮本機。

    RandomizedSearch 會從以下分布抽樣：
    """
    return {
        "n_estimators": randint(80, 401),
        "learning_rate": loguniform(0.01, 0.20),
        "max_depth": randint(2, 8),
        "min_child_weight": randint(1, 11),
        "subsample": uniform(0.60, 0.40),
        "colsample_bytree": uniform(0.60, 0.40),
        "gamma": uniform(0.0, 0.50),
        "reg_alpha": loguniform(1e-4, 2.0),
        "reg_lambda": loguniform(0.10, 10.0),
    }


def evaluate_model(
    name: str,
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_holdout)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    clipped = np.clip(
        probabilities,
        1e-6,
        1.0 - 1e-6,
    )

    summary = {
        "model": name,
        "samples": len(y_holdout),
        "accuracy": accuracy_score(
            y_holdout,
            predictions,
        ),
        "precision": precision_score(
            y_holdout,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_holdout,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_holdout,
            predictions,
            zero_division=0,
        ),
        "brier_score": brier_score_loss(
            y_holdout,
            probabilities,
        ),
        "log_loss": log_loss(
            y_holdout,
            clipped,
            labels=[0, 1],
        ),
        "average_confidence": np.maximum(
            probabilities,
            1.0 - probabilities,
        ).mean(),
        "predicted_up_rate": predictions.mean(),
        "actual_up_rate": y_holdout.mean(),
    }

    predictions_df = pd.DataFrame(
        {
            "model": name,
            "actual": y_holdout.to_numpy(),
            "prediction": predictions,
            "up_probability": probabilities,
            "confidence": np.maximum(
                probabilities,
                1.0 - probabilities,
            ),
            "correct": (
                predictions == y_holdout.to_numpy()
            ).astype(int),
        }
    )

    return summary, predictions_df


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    best_score: float,
    best_params: dict[str, Any],
) -> None:
    print()
    print("=" * 82)
    print("V10.3 XGBoost Hyperparameter Search")
    print("=" * 82)
    print(
        "搜尋訓練區間："
        f"{train_df.iloc[0]['date'].date()} "
        f"至 {train_df.iloc[-1]['date'].date()}，"
        f"{len(train_df)} 筆"
    )
    print(
        "Final Holdout："
        f"{holdout_df.iloc[0]['date'].date()} "
        f"至 {holdout_df.iloc[-1]['date'].date()}，"
        f"{len(holdout_df)} 筆"
    )
    print(f"TimeSeriesSplit 最佳平均 F1：{best_score:.4f}")
    print("-" * 82)
    print("最佳參數：")
    print(json.dumps(
        best_params,
        ensure_ascii=False,
        indent=2,
    ))
    print("-" * 82)

    printable = summary_df.copy()

    for column in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "average_confidence",
        "predicted_up_rate",
        "actual_up_rate",
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


def run_search(
    stock_id: str,
    start_date: str | None,
    n_iter: int,
    cv_splits: int,
    holdout_ratio: float,
    minimum_holdout_samples: int,
) -> None:
    stock_code = resolve_stock_code(stock_id)

    print("=" * 82)
    print("V10.3 Hyperparameter Search")
    print("=" * 82)
    print(f"股票代號：{stock_code}")
    print("正在建立 Feature Data……")

    raw_df = build_feature_data(stock_code)

    model_df = prepare_data(
        raw_df,
        start_date,
    )

    train_df, holdout_df = split_train_holdout(
        model_df,
        holdout_ratio,
        minimum_holdout_samples,
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["Target"]

    X_holdout = holdout_df[FEATURE_COLUMNS]
    y_holdout = holdout_df["Target"]

    time_series_cv = TimeSeriesSplit(
        n_splits=cv_splits,
    )

    search = RandomizedSearchCV(
        estimator=create_search_model(),
        param_distributions=build_parameter_distributions(),
        n_iter=n_iter,
        scoring="f1",
        cv=time_series_cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
        refit=True,
    )

    print(
        f"開始搜尋：{n_iter} 組參數，"
        f"{cv_splits} 組 TimeSeriesSplit"
    )

    search.fit(
        X_train,
        y_train,
    )

    best_params = {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in search.best_params_.items()
    }

    baseline_summary, baseline_predictions = evaluate_model(
        "Current Baseline",
        create_baseline_model(),
        X_train,
        y_train,
        X_holdout,
        y_holdout,
    )

    best_model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    best_summary, best_predictions = evaluate_model(
        "RandomizedSearch Best",
        best_model,
        X_train,
        y_train,
        X_holdout,
        y_holdout,
    )

    summary_df = pd.DataFrame(
        [
            baseline_summary,
            best_summary,
        ]
    )

    holdout_dates = holdout_df[
        ["date"]
    ].reset_index(drop=True)

    prediction_frames = []

    for predictions_df in [
        baseline_predictions,
        best_predictions,
    ]:
        current = pd.concat(
            [
                holdout_dates,
                predictions_df.reset_index(drop=True),
            ],
            axis=1,
        )

        prediction_frames.append(current)

    all_predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_stock_code = stock_code.replace(".", "_")

    summary_path = (
        results_dir
        / f"hyperparameter_{safe_stock_code}_summary.csv"
    )

    params_path = (
        results_dir
        / f"hyperparameter_{safe_stock_code}_best_params.json"
    )

    search_results_path = (
        results_dir
        / f"hyperparameter_{safe_stock_code}_search_results.csv"
    )

    predictions_path = (
        results_dir
        / f"hyperparameter_{safe_stock_code}_holdout_predictions.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    with params_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "stock_code": stock_code,
                "best_cv_f1": float(search.best_score_),
                "best_params": best_params,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    search_results_df = pd.DataFrame(
        search.cv_results_
    ).sort_values(
        "rank_test_score"
    )

    useful_search_columns = [
        column
        for column in search_results_df.columns
        if (
            column.startswith("param_")
            or column in {
                "rank_test_score",
                "mean_test_score",
                "std_test_score",
                "mean_fit_time",
            }
        )
    ]

    search_results_df[
        useful_search_columns
    ].to_csv(
        search_results_path,
        index=False,
        encoding="utf-8-sig",
    )

    all_predictions_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(
        train_df,
        holdout_df,
        summary_df,
        float(search.best_score_),
        best_params,
    )

    print()
    print("=" * 82)
    print("輸出完成")
    print("=" * 82)
    print(f"模型比較摘要：{summary_path}")
    print(f"最佳參數：{params_path}")
    print(f"搜尋完整結果：{search_results_path}")
    print(f"Holdout 預測：{predictions_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V10.3 XGBoost TimeSeriesSplit "
            "RandomizedSearch"
        )
    )

    parser.add_argument(
        "stock_id",
        help="股票代號，例如 2330 或 2330.TW",
    )

    parser.add_argument(
        "--start-date",
        default=None,
        help="可選資料起始日，例如 2020-01-01",
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=40,
        help="隨機搜尋組數，預設 40",
    )

    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="TimeSeriesSplit 折數，預設 5",
    )

    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.20,
        help="最後保留測試資料比例，預設 0.20",
    )

    parser.add_argument(
        "--minimum-holdout-samples",
        type=int,
        default=120,
        help="Final Holdout 最少樣本數，預設 120",
    )

    args = parser.parse_args()

    if args.n_iter < 1:
        parser.error("--n-iter 必須至少為 1")

    if args.cv_splits < 3:
        parser.error("--cv-splits 必須至少為 3")

    if not 0.10 <= args.holdout_ratio <= 0.40:
        parser.error(
            "--holdout-ratio 必須介於 0.10 到 0.40"
        )

    return args


def main() -> None:
    args = parse_args()

    run_search(
        stock_id=args.stock_id,
        start_date=args.start_date,
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        holdout_ratio=args.holdout_ratio,
        minimum_holdout_samples=args.minimum_holdout_samples,
    )


if __name__ == "__main__":
    main()
