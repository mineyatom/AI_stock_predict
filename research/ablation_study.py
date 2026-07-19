"""
V10.6 Backward Ablation Study

用途：
- 不修改正式 predictor.py。
- 以 V10.5 選出的 Baseline + ROC 為完整模型。
- 使用 Expanding Window Walk-Forward，逐輪嘗試移除 Baseline 特徵。
- 所有實驗共用完全相同的日期與樣本。

快速篩選：
    D:\\conda_envs\\stock_ai\\python.exe research\\ablation_study.py 2330 --step 5

正式逐日驗證：
    D:\\conda_envs\\stock_ai\\python.exe research\\ablation_study.py 2330 --step 1
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "V10.6.0-backward-ablation"
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
        spec = importlib.util.spec_from_file_location("v10_feature_lab_shared", module_path)
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
        ]
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise ImportError(f"{module_path.name} 缺少：{', '.join(missing)}")
        print(f"共用模組：{module_path.name}")
        return module
    raise FileNotFoundError("找不到 feature_lab_v10_4_1.py 或 feature_lab.py")


def build_common_dataframe(
    lab: Any,
    stock_code: str,
    start_date: str | None,
    selected_features: list[str],
    minimum_train_samples: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    raw_df = lab.build_feature_data(stock_code)
    df = lab.prepare_base_dataframe(raw_df, start_date)

    selected_groups: dict[str, list[str]] = {}
    for name in selected_features:
        if name not in lab.FEATURE_BUILDERS:
            raise ValueError(f"未知新增特徵：{name}")
        builder, columns = lab.FEATURE_BUILDERS[name]
        df = builder(df)
        selected_groups[name] = list(columns)

    all_columns = list(lab.BASELINE_FEATURES)
    for cols in selected_groups.values():
        all_columns.extend(cols)

    required = ["date", "Target", *all_columns]
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
            f"共同有效資料不足：{len(common_df)} 筆；"
            f"minimum_train_samples={minimum_train_samples}"
        )

    print(
        f"共同實驗資料：{common_df.iloc[0]['date'].date()} 至 "
        f"{common_df.iloc[-1]['date'].date()}，共 {len(common_df)} 筆"
    )
    return common_df, selected_groups


def metric_deltas(candidate: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    return {
        "accuracy_delta": candidate["accuracy"] - current["accuracy"],
        "precision_delta": candidate["precision"] - current["precision"],
        "recall_delta": candidate["recall"] - current["recall"],
        "f1_delta": candidate["f1"] - current["f1"],
        "brier_delta": candidate["brier_score"] - current["brier_score"],
        "log_loss_delta": candidate["log_loss"] - current["log_loss"],
    }


def decide_removal(
    deltas: dict[str, float],
    max_accuracy_drop: float,
    max_f1_drop: float,
    max_brier_worsening: float,
    max_log_loss_worsening: float,
) -> tuple[bool, str]:
    if deltas["accuracy_delta"] < -max_accuracy_drop:
        return False, f"Accuracy 下降超過 {max_accuracy_drop:.2%}"
    if deltas["f1_delta"] < -max_f1_drop:
        return False, f"F1 下降超過 {max_f1_drop:.2%}"
    if deltas["brier_delta"] > max_brier_worsening:
        return False, f"Brier 惡化超過 {max_brier_worsening:.6f}"
    if deltas["log_loss_delta"] > max_log_loss_worsening:
        return False, f"LogLoss 惡化超過 {max_log_loss_worsening:.6f}"
    return True, "可移除"


def removal_rank(row: dict[str, Any]) -> tuple[float, ...]:
    """優先移除 Accuracy/F1 改善最多且機率品質更好的特徵。"""
    return (
        float(row["accuracy_delta"]),
        float(row["f1_delta"]),
        -float(row["brier_delta"]),
        -float(row["log_loss_delta"]),
    )


def feature_set_label(features: list[str], additions: list[str]) -> str:
    base = ", ".join(features) if features else "None"
    extra = " + ".join(additions)
    return f"[{base}] + {extra}" if extra else f"[{base}]"


def run_ablation(args: argparse.Namespace) -> None:
    lab = load_feature_lab_module()
    stock_code = lab.resolve_stock_code(args.stock_id)
    additions = [name.upper() for name in args.selected_features]

    print("=" * 96)
    print(f"V10.6 Backward Ablation Study｜{SCRIPT_VERSION}")
    print("=" * 96)
    print(f"股票代號：{stock_code}")
    print(f"固定新增特徵：{', '.join(additions)}")
    print(f"Walk-Forward step：{args.step}")

    common_df, selected_groups = build_common_dataframe(
        lab,
        stock_code,
        args.start_date,
        additions,
        args.minimum_train_samples,
    )

    fixed_columns: list[str] = []
    for name in additions:
        fixed_columns.extend(selected_groups[name])

    remaining = list(lab.BASELINE_FEATURES)
    full_columns = [*remaining, *fixed_columns]
    current_metrics, current_predictions = lab.walk_forward_predict(
        common_df,
        full_columns,
        args.minimum_train_samples,
        args.step,
        "Round_0_Full_Model",
    )

    original_metrics = dict(current_metrics)
    removed: list[str] = []
    path_rows: list[dict[str, Any]] = [{
        "round": 0,
        "removed_feature": "None",
        "remaining_baseline_features": ",".join(remaining),
        "fixed_additions": ",".join(additions),
        "feature_count": len(full_columns),
        **current_metrics,
        "accuracy_delta_from_previous": 0.0,
        "f1_delta_from_previous": 0.0,
        "brier_delta_from_previous": 0.0,
        "log_loss_delta_from_previous": 0.0,
    }]
    candidate_rows: list[dict[str, Any]] = []
    accepted_predictions = [current_predictions.assign(ablation_round=0)]

    max_rounds = min(args.max_rounds, len(remaining) - args.minimum_baseline_features)

    for round_number in range(1, max_rounds + 1):
        if len(remaining) <= args.minimum_baseline_features:
            break

        print("\n" + "-" * 96)
        print(f"Round {round_number}｜目前剩餘 Baseline 特徵：{len(remaining)}")
        print("-" * 96)

        round_rows: list[dict[str, Any]] = []
        round_predictions: dict[str, pd.DataFrame] = {}

        for feature in remaining:
            trial_remaining = [name for name in remaining if name != feature]
            trial_columns = [*trial_remaining, *fixed_columns]
            metrics, predictions = lab.walk_forward_predict(
                common_df,
                trial_columns,
                args.minimum_train_samples,
                args.step,
                f"Round_{round_number}_Remove_{feature}",
            )
            deltas = metric_deltas(metrics, current_metrics)
            accepted, reason = decide_removal(
                deltas,
                args.max_accuracy_drop,
                args.max_f1_drop,
                args.max_brier_worsening,
                args.max_log_loss_worsening,
            )
            row = {
                "round": round_number,
                "candidate_removed": feature,
                "remaining_baseline_features": ",".join(trial_remaining),
                "feature_count": len(trial_columns),
                **metrics,
                **deltas,
                "passes_threshold": accepted,
                "decision_reason": reason,
            }
            round_rows.append(row)
            candidate_rows.append(row.copy())
            round_predictions[feature] = predictions

            print(
                f"移除 {feature:<20} "
                f"Accuracy={metrics['accuracy']:.2%} ({deltas['accuracy_delta']:+.2%})｜"
                f"F1={metrics['f1']:.2%} ({deltas['f1_delta']:+.2%})｜"
                f"Brier Δ={deltas['brier_delta']:+.6f}｜"
                f"LogLoss Δ={deltas['log_loss_delta']:+.6f}｜"
                f"{'PASS' if accepted else 'KEEP'}"
            )

        eligible = [row for row in round_rows if row["passes_threshold"]]
        if not eligible:
            print("本輪沒有任何 Baseline 特徵可安全移除，消融結束。")
            break

        winner = max(eligible, key=removal_rank)
        feature = str(winner["candidate_removed"])
        remaining.remove(feature)
        removed.append(feature)

        current_metrics = {key: winner[key] for key in original_metrics.keys()}
        current_predictions = round_predictions[feature].copy()
        current_predictions["ablation_round"] = round_number
        accepted_predictions.append(current_predictions)

        path_rows.append({
            "round": round_number,
            "removed_feature": feature,
            "remaining_baseline_features": ",".join(remaining),
            "fixed_additions": ",".join(additions),
            "feature_count": len(remaining) + len(fixed_columns),
            **current_metrics,
            "accuracy_delta_from_previous": winner["accuracy_delta"],
            "f1_delta_from_previous": winner["f1_delta"],
            "brier_delta_from_previous": winner["brier_delta"],
            "log_loss_delta_from_previous": winner["log_loss_delta"],
        })
        print(f"本輪正式移除：{feature}")

    results_dir = Path(args.output_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_stock = stock_code.replace(".", "_")
    prefix = results_dir / f"ablation_{safe_stock}"
    path_file = Path(f"{prefix}_path.csv")
    candidates_file = Path(f"{prefix}_round_candidates.csv")
    predictions_file = Path(f"{prefix}_predictions.csv")
    report_file = Path(f"{prefix}_report.txt")

    pd.DataFrame(path_rows).to_csv(path_file, index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_rows).to_csv(candidates_file, index=False, encoding="utf-8-sig")
    pd.concat(accepted_predictions, ignore_index=True).to_csv(
        predictions_file, index=False, encoding="utf-8-sig"
    )

    final_metrics = current_metrics
    report_lines = [
        "V10.6 Backward Ablation Study Report",
        "=" * 64,
        f"Script Version: {SCRIPT_VERSION}",
        f"Stock: {stock_code}",
        f"Date Range: {common_df.iloc[0]['date'].date()} ~ {common_df.iloc[-1]['date'].date()}",
        f"Common Rows: {len(common_df)}",
        f"Minimum Train Samples: {args.minimum_train_samples}",
        f"Walk-Forward Step: {args.step}",
        f"Fixed Additions: {', '.join(additions)}",
        "",
        f"Original Baseline Features ({len(lab.BASELINE_FEATURES)}): {', '.join(lab.BASELINE_FEATURES)}",
        f"Removed Baseline Features ({len(removed)}): {', '.join(removed) if removed else 'None'}",
        f"Remaining Baseline Features ({len(remaining)}): {', '.join(remaining)}",
        f"Final Feature Set: {feature_set_label(remaining, additions)}",
        "",
        f"Original Accuracy: {original_metrics['accuracy']:.4%}",
        f"Final Accuracy: {final_metrics['accuracy']:.4%}",
        f"Accuracy Change: {final_metrics['accuracy'] - original_metrics['accuracy']:+.4%}",
        f"Original F1: {original_metrics['f1']:.4%}",
        f"Final F1: {final_metrics['f1']:.4%}",
        f"F1 Change: {final_metrics['f1'] - original_metrics['f1']:+.4%}",
        f"Original Brier: {original_metrics['brier_score']:.6f}",
        f"Final Brier: {final_metrics['brier_score']:.6f}",
        f"Original LogLoss: {original_metrics['log_loss']:.6f}",
        f"Final LogLoss: {final_metrics['log_loss']:.6f}",
        "",
        "Ablation Path:",
    ]
    for row in path_rows:
        report_lines.append(
            f"Round {row['round']}: removed={row['removed_feature']} | "
            f"features={row['feature_count']} | Accuracy={row['accuracy']:.4%} | "
            f"F1={row['f1']:.4%} | Brier={row['brier_score']:.6f} | "
            f"LogLoss={row['log_loss']:.6f}"
        )
    report_lines.extend([
        "",
        "Removal Rules:",
        f"- Maximum accuracy drop: {args.max_accuracy_drop:.4%}",
        f"- Maximum F1 drop: {args.max_f1_drop:.4%}",
        f"- Maximum Brier worsening: {args.max_brier_worsening:.6f}",
        f"- Maximum LogLoss worsening: {args.max_log_loss_worsening:.6f}",
        "",
        "Important: --step 5 is a screening result. Use --step 1 before production integration.",
    ])
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 96)
    print("V10.6 Ablation Result")
    print("=" * 96)
    print(f"移除：{', '.join(removed) if removed else '無'}")
    print(f"保留 Baseline：{', '.join(remaining)}")
    print(f"固定新增：{', '.join(additions)}")
    print(f"Accuracy：{original_metrics['accuracy']:.2%} → {final_metrics['accuracy']:.2%}")
    print(f"F1：{original_metrics['f1']:.2%} → {final_metrics['f1']:.2%}")
    print("輸出檔案：")
    print(f"- {path_file}")
    print(f"- {candidates_file}")
    print(f"- {predictions_file}")
    print(f"- {report_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V10.6 自動後向消融研究")
    parser.add_argument("stock_id", help="股票代號，例如 2330 或 2330.TW")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--minimum-train-samples", type=int, default=252)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument(
        "--selected-features",
        nargs="+",
        default=["ROC"],
        help="V10.5 已選入並固定保留的新增特徵，預設 ROC",
    )
    parser.add_argument("--max-rounds", type=int, default=17)
    parser.add_argument("--minimum-baseline-features", type=int, default=5)
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=0.0,
        help="允許移除後 Accuracy 最大下降，預設 0（不可下降）",
    )
    parser.add_argument(
        "--max-f1-drop",
        type=float,
        default=0.005,
        help="允許 F1 最大下降，預設 0.005（0.5 個百分點）",
    )
    parser.add_argument("--max-brier-worsening", type=float, default=0.002)
    parser.add_argument("--max-log-loss-worsening", type=float, default=0.01)
    parser.add_argument("--output-dir", default="research/results")
    return parser.parse_args()


def main() -> None:
    run_ablation(parse_args())


if __name__ == "__main__":
    main()
