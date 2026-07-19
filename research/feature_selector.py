"""
V10.5 Forward Feature Selection

用途：
- 不修改正式 predictor.py。
- 沿用 V10.4 Feature Lab 的資料處理、候選指標與 XGBoost 參數。
- 使用 Expanding Window Walk-Forward 自動執行前向特徵選擇。
- 所有回合與候選組合使用完全相同的共同有效日期，確保公平比較。

執行：
    D:\\conda_envs\\stock_ai\\python.exe research\\feature_selector.py 2330 --step 5

正式逐日驗證：
    D:\\conda_envs\\stock_ai\\python.exe research\\feature_selector.py 2330 --step 1

輸出：
    research/results/feature_selector_2330_TW_selection_path.csv
    research/results/feature_selector_2330_TW_round_candidates.csv
    research/results/feature_selector_2330_TW_predictions.csv
    research/results/feature_selector_2330_TW_report.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "V10.5.0-forward-selection"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_feature_lab_module() -> Any:
    """優先載入修正版 Feature Lab，找不到時再載入 feature_lab.py。"""
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

        required_names = [
            "BASELINE_FEATURES",
            "FEATURE_BUILDERS",
            "build_feature_data",
            "resolve_stock_code",
            "prepare_base_dataframe",
            "walk_forward_predict",
            "calculate_metrics",
        ]
        missing = [name for name in required_names if not hasattr(module, name)]
        if missing:
            raise ImportError(
                f"{module_path.name} 缺少必要內容：{', '.join(missing)}"
            )

        print(f"共用模組：{module_path.name}")
        return module

    raise FileNotFoundError(
        "找不到 research/feature_lab_v10_4_1.py 或 research/feature_lab.py。"
    )


def normalize_feature_names(
    requested: list[str] | None,
    available: list[str],
) -> list[str]:
    if not requested:
        return available

    normalized = []
    for name in requested:
        upper_name = name.strip().upper()
        if upper_name not in available:
            raise ValueError(
                f"未知候選特徵：{name}；可用值：{', '.join(available)}"
            )
        if upper_name not in normalized:
            normalized.append(upper_name)

    return normalized


def build_common_dataframe(
    lab: Any,
    stock_code: str,
    start_date: str | None,
    candidate_names: list[str],
    minimum_train_samples: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    print("正在建立正式 Feature Data……")
    raw_df = lab.build_feature_data(stock_code)
    base_df = lab.prepare_base_dataframe(raw_df, start_date)
    experiment_df = base_df.copy()

    feature_groups: dict[str, list[str]] = {}
    for feature_name in candidate_names:
        builder, new_columns = lab.FEATURE_BUILDERS[feature_name]
        experiment_df = builder(experiment_df)
        feature_groups[feature_name] = list(new_columns)

    all_features = list(lab.BASELINE_FEATURES)
    for columns in feature_groups.values():
        all_features.extend(columns)

    required_columns = ["date", "Target", *all_features]
    common_df = (
        experiment_df[required_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required_columns)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    if len(common_df) <= minimum_train_samples:
        raise ValueError(
            "所有候選特徵共同有效資料不足："
            f"{len(common_df)} 筆；minimum_train_samples={minimum_train_samples}。"
        )

    print(
        "共同實驗資料："
        f"{common_df.iloc[0]['date'].date()} 至 "
        f"{common_df.iloc[-1]['date'].date()}，共 {len(common_df)} 筆"
    )
    return common_df, feature_groups


def metric_deltas(
    candidate: dict[str, float],
    current: dict[str, float],
) -> dict[str, float]:
    return {
        "accuracy_delta": candidate["accuracy"] - current["accuracy"],
        "precision_delta": candidate["precision"] - current["precision"],
        "recall_delta": candidate["recall"] - current["recall"],
        "f1_delta": candidate["f1"] - current["f1"],
        "brier_delta": candidate["brier_score"] - current["brier_score"],
        "log_loss_delta": candidate["log_loss"] - current["log_loss"],
    }


def decide_acceptance(
    deltas: dict[str, float],
    min_accuracy_gain: float,
    max_brier_worsening: float,
    max_log_loss_worsening: float,
) -> tuple[bool, str, int]:
    improved_count = sum(
        [
            deltas["accuracy_delta"] > 0,
            deltas["f1_delta"] > 0,
            deltas["brier_delta"] < 0,
            deltas["log_loss_delta"] < 0,
        ]
    )

    if deltas["accuracy_delta"] < min_accuracy_gain:
        return (
            False,
            f"Accuracy 提升不足 {min_accuracy_gain:.2%}",
            improved_count,
        )

    if deltas["brier_delta"] > max_brier_worsening:
        return (
            False,
            f"Brier 惡化超過 {max_brier_worsening:.6f}",
            improved_count,
        )

    if deltas["log_loss_delta"] > max_log_loss_worsening:
        return (
            False,
            f"LogLoss 惡化超過 {max_log_loss_worsening:.6f}",
            improved_count,
        )

    return True, "通過", improved_count


def candidate_rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Accuracy 優先，其次 F1，再偏好較低 Brier 與 LogLoss。"""
    return (
        float(row["accuracy_delta"]),
        float(row["f1_delta"]),
        -float(row["brier_delta"]),
        -float(row["log_loss_delta"]),
    )


def format_feature_set(selected: list[str]) -> str:
    return "Baseline" if not selected else "Baseline + " + " + ".join(selected)


def run_forward_selection(args: argparse.Namespace) -> None:
    lab = load_feature_lab_module()
    available = list(lab.FEATURE_BUILDERS.keys())
    candidates = normalize_feature_names(args.features, available)
    stock_code = lab.resolve_stock_code(args.stock_id)

    print("=" * 92)
    print(f"V10.5 Forward Feature Selection｜{SCRIPT_VERSION}")
    print("=" * 92)
    print(f"股票代號：{stock_code}")
    print(f"候選特徵：{', '.join(candidates)}")
    print(f"Walk-Forward step：{args.step}")
    print(f"最低 Accuracy 改善：{args.min_accuracy_gain:.2%}")

    common_df, feature_groups = build_common_dataframe(
        lab=lab,
        stock_code=stock_code,
        start_date=args.start_date,
        candidate_names=candidates,
        minimum_train_samples=args.minimum_train_samples,
    )

    baseline_metrics, baseline_predictions = lab.walk_forward_predict(
        common_df,
        list(lab.BASELINE_FEATURES),
        args.minimum_train_samples,
        args.step,
        "Round_0_Baseline",
    )

    current_metrics = baseline_metrics
    current_predictions = baseline_predictions.copy()
    selected: list[str] = []
    remaining = list(candidates)

    selection_rows: list[dict[str, Any]] = [
        {
            "round": 0,
            "selected_feature": "Baseline",
            "selected_features": "",
            "feature_set": "Baseline",
            **baseline_metrics,
            "accuracy_delta_from_previous": 0.0,
            "f1_delta_from_previous": 0.0,
            "brier_delta_from_previous": 0.0,
            "log_loss_delta_from_previous": 0.0,
        }
    ]
    candidate_rows: list[dict[str, Any]] = []
    accepted_prediction_frames = [baseline_predictions.assign(selection_round=0)]

    max_rounds = min(args.max_rounds, len(candidates))

    for round_number in range(1, max_rounds + 1):
        if not remaining:
            break

        print()
        print("-" * 92)
        print(
            f"Round {round_number}｜目前組合：{format_feature_set(selected)}｜"
            f"剩餘 {len(remaining)} 個候選"
        )
        print("-" * 92)

        round_results: list[dict[str, Any]] = []
        round_predictions: dict[str, pd.DataFrame] = {}

        for candidate in remaining:
            trial_selected = [*selected, candidate]
            feature_columns = list(lab.BASELINE_FEATURES)
            for feature_name in trial_selected:
                feature_columns.extend(feature_groups[feature_name])

            experiment_name = f"Round_{round_number}_{'_'.join(trial_selected)}"
            metrics, predictions = lab.walk_forward_predict(
                common_df,
                feature_columns,
                args.minimum_train_samples,
                args.step,
                experiment_name,
            )
            deltas = metric_deltas(metrics, current_metrics)
            accepted, reason, improved_count = decide_acceptance(
                deltas,
                args.min_accuracy_gain,
                args.max_brier_worsening,
                args.max_log_loss_worsening,
            )

            row = {
                "round": round_number,
                "candidate": candidate,
                "trial_selected_features": ",".join(trial_selected),
                "feature_set": format_feature_set(trial_selected),
                **metrics,
                **deltas,
                "improved_metric_count": improved_count,
                "passes_threshold": accepted,
                "decision_reason": reason,
            }
            round_results.append(row)
            candidate_rows.append(row.copy())
            round_predictions[candidate] = predictions

            print(
                f"{candidate:<10} "
                f"Accuracy={metrics['accuracy']:.2%} "
                f"(Δ {deltas['accuracy_delta']:+.2%})｜"
                f"F1={metrics['f1']:.2%} "
                f"(Δ {deltas['f1_delta']:+.2%})｜"
                f"Brier Δ={deltas['brier_delta']:+.6f}｜"
                f"LogLoss Δ={deltas['log_loss_delta']:+.6f}｜"
                f"{'PASS' if accepted else 'REJECT'}"
            )

        eligible = [row for row in round_results if row["passes_threshold"]]
        if not eligible:
            print("本輪沒有候選特徵通過停止條件，前向選擇結束。")
            break

        winner = max(eligible, key=candidate_rank_key)
        winner_name = str(winner["candidate"])
        selected.append(winner_name)
        remaining.remove(winner_name)

        current_metrics = {
            key: winner[key]
            for key in baseline_metrics.keys()
        }
        current_predictions = round_predictions[winner_name].copy()
        current_predictions["selection_round"] = round_number
        accepted_prediction_frames.append(current_predictions)

        selection_rows.append(
            {
                "round": round_number,
                "selected_feature": winner_name,
                "selected_features": ",".join(selected),
                "feature_set": format_feature_set(selected),
                **current_metrics,
                "accuracy_delta_from_previous": winner["accuracy_delta"],
                "f1_delta_from_previous": winner["f1_delta"],
                "brier_delta_from_previous": winner["brier_delta"],
                "log_loss_delta_from_previous": winner["log_loss_delta"],
            }
        )

        print(
            f"本輪選入：{winner_name}｜"
            f"新組合 Accuracy={current_metrics['accuracy']:.2%}，"
            f"F1={current_metrics['f1']:.2%}"
        )

    results_dir = Path(args.output_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_stock = stock_code.replace(".", "_")
    prefix = results_dir / f"feature_selector_{safe_stock}"

    selection_df = pd.DataFrame(selection_rows)
    candidates_df = pd.DataFrame(candidate_rows)
    predictions_df = pd.concat(accepted_prediction_frames, ignore_index=True)

    selection_path = Path(f"{prefix}_selection_path.csv")
    round_candidates_path = Path(f"{prefix}_round_candidates.csv")
    predictions_path = Path(f"{prefix}_predictions.csv")
    report_path = Path(f"{prefix}_report.txt")

    selection_df.to_csv(selection_path, index=False, encoding="utf-8-sig")
    candidates_df.to_csv(round_candidates_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    baseline_accuracy = float(selection_df.iloc[0]["accuracy"])
    final_accuracy = float(selection_df.iloc[-1]["accuracy"])
    baseline_f1 = float(selection_df.iloc[0]["f1"])
    final_f1 = float(selection_df.iloc[-1]["f1"])

    report_lines = [
        "V10.5 Forward Feature Selection Report",
        "=" * 60,
        f"Script Version: {SCRIPT_VERSION}",
        f"Stock: {stock_code}",
        f"Date Range: {common_df.iloc[0]['date'].date()} ~ {common_df.iloc[-1]['date'].date()}",
        f"Common Rows: {len(common_df)}",
        f"Minimum Train Samples: {args.minimum_train_samples}",
        f"Walk-Forward Step: {args.step}",
        f"Candidates: {', '.join(candidates)}",
        "",
        f"Selected Features: {', '.join(selected) if selected else 'None'}",
        f"Final Feature Set: {format_feature_set(selected)}",
        f"Baseline Accuracy: {baseline_accuracy:.4%}",
        f"Final Accuracy: {final_accuracy:.4%}",
        f"Accuracy Improvement: {final_accuracy - baseline_accuracy:+.4%}",
        f"Baseline F1: {baseline_f1:.4%}",
        f"Final F1: {final_f1:.4%}",
        f"F1 Improvement: {final_f1 - baseline_f1:+.4%}",
        "",
        "Selection Path:",
    ]

    for row in selection_rows:
        report_lines.append(
            f"Round {row['round']}: {row['feature_set']} | "
            f"Accuracy={row['accuracy']:.4%} | "
            f"F1={row['f1']:.4%} | "
            f"Brier={row['brier_score']:.6f} | "
            f"LogLoss={row['log_loss']:.6f}"
        )

    report_lines.extend(
        [
            "",
            "Stopping Rules:",
            f"- Minimum accuracy gain: {args.min_accuracy_gain:.4%}",
            f"- Maximum Brier worsening: {args.max_brier_worsening:.6f}",
            f"- Maximum LogLoss worsening: {args.max_log_loss_worsening:.6f}",
            "",
            "Important: --step 5 is only a fast screening result. "
            "Before production integration, rerun with --step 1 and perform ablation study.",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print()
    print("=" * 92)
    print("V10.5 Selection Result")
    print("=" * 92)
    print(f"最終選中特徵：{', '.join(selected) if selected else '無'}")
    print(f"最終組合：{format_feature_set(selected)}")
    print(f"Accuracy：{baseline_accuracy:.2%} → {final_accuracy:.2%}")
    print(f"F1：{baseline_f1:.2%} → {final_f1:.2%}")
    print("輸出檔案：")
    print(f"- {selection_path}")
    print(f"- {round_candidates_path}")
    print(f"- {predictions_path}")
    print(f"- {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V10.5 自動前向特徵選擇器"
    )
    parser.add_argument("stock_id", help="股票代號，例如 2330 或 2330.TW")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--minimum-train-samples", type=int, default=252)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="指定候選特徵，例如 --features ROC MFI ADX OBV",
    )
    parser.add_argument("--max-rounds", type=int, default=7)
    parser.add_argument(
        "--min-accuracy-gain",
        type=float,
        default=0.0025,
        help="每輪最低 Accuracy 提升，0.0025 = 0.25 個百分點",
    )
    parser.add_argument("--max-brier-worsening", type=float, default=0.002)
    parser.add_argument("--max-log-loss-worsening", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        default="research/results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_forward_selection(args)


if __name__ == "__main__":
    main()
