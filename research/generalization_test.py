"""
V10.7 Multi-Stock Generalization Test

正式驗證：
    D:\\conda_envs\\stock_ai\\python.exe research\\generalization_test.py 2454

快速測試：
    D:\\conda_envs\\stock_ai\\python.exe research\\generalization_test.py 2454 --step 5

更換候選特徵：
    D:\\conda_envs\\stock_ai\\python.exe research\\generalization_test.py 2454 --candidate MFI
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "V10.7.0-multi-stock-generalization"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_feature_lab_module() -> Any:
    candidates = [
        RESEARCH_DIR / "feature_lab_v10_4_1.py",
        RESEARCH_DIR / "feature_lab.py",
    ]

    for module_path in candidates:
        if not module_path.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            "v10_feature_lab_shared",
            module_path,
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        required = [
            "BASELINE_FEATURES",
            "FEATURE_BUILDERS",
            "build_feature_data",
            "resolve_stock_code",
            "prepare_base_dataframe",
            "walk_forward_predict",
            "calculate_metrics",
        ]
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise ImportError(
                f"{module_path.name} 缺少必要內容：{', '.join(missing)}"
            )

        print(f"共用模組：{module_path.name}")
        return module

    raise FileNotFoundError(
        "找不到 research/feature_lab_v10_4_1.py 或 research/feature_lab.py"
    )


def build_common_dataframe(
    lab: Any,
    stock_code: str,
    candidate: str,
    start_date: str | None,
    minimum_train_samples: int,
) -> tuple[pd.DataFrame, list[str]]:
    raw_df = lab.build_feature_data(stock_code)
    df = lab.prepare_base_dataframe(raw_df, start_date)

    builder, candidate_columns = lab.FEATURE_BUILDERS[candidate]
    df = builder(df)
    candidate_columns = list(candidate_columns)

    required = [
        "date",
        "Target",
        *lab.BASELINE_FEATURES,
        *candidate_columns,
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("實驗資料缺少必要欄位：" + ", ".join(missing))

    common_df = (
        df[required]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    if len(common_df) <= minimum_train_samples:
        raise ValueError(
            f"共同有效資料只有 {len(common_df)} 筆，"
            f"不足 minimum_train_samples={minimum_train_samples}"
        )

    print(
        f"共同實驗資料：{common_df.iloc[0]['date'].date()} 至 "
        f"{common_df.iloc[-1]['date'].date()}，共 {len(common_df)} 筆"
    )
    return common_df, candidate_columns


def align_predictions(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baseline.copy()
    candidate = candidate.copy()

    baseline["date"] = pd.to_datetime(
        baseline["date"], errors="coerce"
    ).dt.normalize()
    candidate["date"] = pd.to_datetime(
        candidate["date"], errors="coerce"
    ).dt.normalize()

    common_dates = sorted(
        set(baseline["date"].dropna())
        & set(candidate["date"].dropna())
    )
    if not common_dates:
        raise ValueError("Baseline 與 Candidate 沒有共同預測日期")

    baseline = (
        baseline[baseline["date"].isin(common_dates)]
        .sort_values("date")
        .reset_index(drop=True)
    )
    candidate = (
        candidate[candidate["date"].isin(common_dates)]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(baseline) != len(candidate):
        raise ValueError("共同日期對齊後樣本數仍不一致")

    return baseline, candidate


def calculate_metrics(lab: Any, predictions: pd.DataFrame) -> dict[str, float]:
    return lab.calculate_metrics(
        predictions["actual"].to_numpy(),
        predictions["up_probability"].to_numpy(),
    )


def metric_deltas(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    return {
        "accuracy_delta": candidate["accuracy"] - baseline["accuracy"],
        "precision_delta": candidate["precision"] - baseline["precision"],
        "recall_delta": candidate["recall"] - baseline["recall"],
        "f1_delta": candidate["f1"] - baseline["f1"],
        "brier_delta": candidate["brier_score"] - baseline["brier_score"],
        "log_loss_delta": candidate["log_loss"] - baseline["log_loss"],
    }


def classify_result(deltas: dict[str, float]) -> tuple[str, str]:
    accuracy_better = deltas["accuracy_delta"] > 0
    f1_better = deltas["f1_delta"] > 0
    brier_better = deltas["brier_delta"] < 0
    log_loss_better = deltas["log_loss_delta"] < 0

    if accuracy_better and f1_better and brier_better and log_loss_better:
        return "STRONG PASS", "Accuracy、F1、Brier、LogLoss 全部改善"

    support = sum([f1_better, brier_better, log_loss_better])
    if accuracy_better and support >= 1:
        return "WEAK PASS", "Accuracy 改善，且至少一項其他品質指標改善"

    if not accuracy_better:
        return "FAIL", "Accuracy 未改善"

    return "FAIL", "只有 Accuracy 改善，但其他品質指標全部惡化"


def build_summary(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    candidate_name: str,
    result: str,
) -> pd.DataFrame:
    deltas = metric_deltas(baseline_metrics, candidate_metrics)

    return pd.DataFrame(
        [
            {
                "model": "Baseline",
                **baseline_metrics,
                "accuracy_delta": 0.0,
                "precision_delta": 0.0,
                "recall_delta": 0.0,
                "f1_delta": 0.0,
                "brier_delta": 0.0,
                "log_loss_delta": 0.0,
                "result": "BASELINE",
            },
            {
                "model": f"Baseline + {candidate_name}",
                **candidate_metrics,
                **deltas,
                "result": result,
            },
        ]
    )


def build_report(
    stock_code: str,
    candidate: str,
    candidate_columns: list[str],
    common_df: pd.DataFrame,
    summary: pd.DataFrame,
    result: str,
    reason: str,
    minimum_train_samples: int,
    step: int,
) -> str:
    base = summary.iloc[0]
    test = summary.iloc[1]

    lines = [
        "V10.7 Multi-Stock Generalization Report",
        "=" * 68,
        f"Script Version: {SCRIPT_VERSION}",
        f"Stock: {stock_code}",
        f"Candidate: {candidate}",
        f"Candidate Columns: {', '.join(candidate_columns)}",
        f"Date Range: {common_df.iloc[0]['date'].date()} ~ {common_df.iloc[-1]['date'].date()}",
        f"Common Data Rows: {len(common_df)}",
        f"Out-of-Sample Predictions: {int(base['samples'])}",
        f"Minimum Train Samples: {minimum_train_samples}",
        f"Walk-Forward Step: {step}",
        "",
        "Baseline",
        "-" * 68,
        f"Accuracy: {float(base['accuracy']):.4%}",
        f"Precision: {float(base['precision']):.4%}",
        f"Recall: {float(base['recall']):.4%}",
        f"F1: {float(base['f1']):.4%}",
        f"Brier: {float(base['brier_score']):.6f}",
        f"LogLoss: {float(base['log_loss']):.6f}",
        "",
        f"Baseline + {candidate}",
        "-" * 68,
        f"Accuracy: {float(test['accuracy']):.4%}",
        f"Precision: {float(test['precision']):.4%}",
        f"Recall: {float(test['recall']):.4%}",
        f"F1: {float(test['f1']):.4%}",
        f"Brier: {float(test['brier_score']):.6f}",
        f"LogLoss: {float(test['log_loss']):.6f}",
        "",
        "Metric Changes",
        "-" * 68,
        f"Accuracy Change: {float(test['accuracy_delta']):+.4%}",
        f"Precision Change: {float(test['precision_delta']):+.4%}",
        f"Recall Change: {float(test['recall_delta']):+.4%}",
        f"F1 Change: {float(test['f1_delta']):+.4%}",
        f"Brier Change: {float(test['brier_delta']):+.6f} (negative is better)",
        f"LogLoss Change: {float(test['log_loss_delta']):+.6f} (negative is better)",
        "",
        f"Result: {result}",
        f"Reason: {reason}",
        "",
        "Important: --step 5 is only a screening run. Use --step 1 for the official result.",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    lab = load_feature_lab_module()
    candidate = args.candidate.upper()

    if candidate not in lab.FEATURE_BUILDERS:
        raise ValueError(
            f"不支援的候選特徵：{candidate}；可用："
            + ", ".join(lab.FEATURE_BUILDERS.keys())
        )

    stock_code = lab.resolve_stock_code(args.stock_id)

    print("=" * 92)
    print(f"V10.7 Multi-Stock Generalization Test｜{SCRIPT_VERSION}")
    print("=" * 92)
    print(f"股票代號：{stock_code}")
    print(f"候選特徵：{candidate}")
    print(f"Walk-Forward step：{args.step}")
    print("正在建立正式 Feature Data……")

    common_df, candidate_columns = build_common_dataframe(
        lab,
        stock_code,
        candidate,
        args.start_date,
        args.minimum_train_samples,
    )

    baseline_features = list(lab.BASELINE_FEATURES)
    candidate_features = [*baseline_features, *candidate_columns]

    _, baseline_predictions = lab.walk_forward_predict(
        common_df,
        baseline_features,
        args.minimum_train_samples,
        args.step,
        "Baseline",
    )
    _, candidate_predictions = lab.walk_forward_predict(
        common_df,
        candidate_features,
        args.minimum_train_samples,
        args.step,
        f"Baseline + {candidate}",
    )

    baseline_predictions, candidate_predictions = align_predictions(
        baseline_predictions,
        candidate_predictions,
    )

    baseline_metrics = calculate_metrics(lab, baseline_predictions)
    candidate_metrics = calculate_metrics(lab, candidate_predictions)
    deltas = metric_deltas(baseline_metrics, candidate_metrics)
    result, reason = classify_result(deltas)

    summary = build_summary(
        baseline_metrics,
        candidate_metrics,
        candidate,
        result,
    )

    predictions = pd.concat(
        [baseline_predictions, candidate_predictions],
        ignore_index=True,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_stock = stock_code.replace(".", "_")
    prefix = output_dir / f"generalization_{safe_stock}_{candidate}"

    summary_path = Path(f"{prefix}.csv")
    predictions_path = Path(f"{prefix}_predictions.csv")
    report_path = Path(f"{prefix}_report.txt")

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(
            stock_code,
            candidate,
            candidate_columns,
            common_df,
            summary,
            result,
            reason,
            args.minimum_train_samples,
            args.step,
        ),
        encoding="utf-8",
    )

    base = summary.iloc[0]
    test = summary.iloc[1]

    print("\n" + "=" * 92)
    print("V10.7 結果")
    print("=" * 92)
    print(f"Baseline Accuracy      : {float(base['accuracy']):.4%}")
    print(f"Baseline + {candidate} Accuracy: {float(test['accuracy']):.4%}")
    print(f"Accuracy Δ             : {float(test['accuracy_delta']):+.4%}")
    print(f"F1 Δ                   : {float(test['f1_delta']):+.4%}")
    print(f"Brier Δ                : {float(test['brier_delta']):+.6f}")
    print(f"LogLoss Δ              : {float(test['log_loss_delta']):+.6f}")
    print(f"判定                    : {result}")
    print(f"原因                    : {reason}")
    print("=" * 92)
    print(f"比較結果：{summary_path}")
    print(f"逐日預測：{predictions_path}")
    print(f"文字報告：{report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V10.7 Multi-Stock Generalization Test"
    )
    parser.add_argument("stock_id", help="股票代號，例如 2454 或 2454.TW")
    parser.add_argument("--candidate", default="ROC", help="候選特徵，預設 ROC")
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="資料起始日，預設 2020-01-01",
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
        help="Walk-Forward 間隔，預設 1；快速測試可設 5",
    )
    parser.add_argument(
        "--output-dir",
        default="research/results",
        help="輸出目錄，預設 research/results",
    )

    args = parser.parse_args()
    args.candidate = args.candidate.strip().upper()

    if args.minimum_train_samples < 100:
        parser.error("--minimum-train-samples 建議至少 100")
    if args.step < 1:
        parser.error("--step 必須至少為 1")

    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
