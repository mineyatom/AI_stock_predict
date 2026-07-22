"""
benchmark_report.py
V11 Benchmark Report

負責整理 Benchmark 結果並輸出 CSV
"""

from pathlib import Path
import pandas as pd


OUTPUT_DIR = Path("benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)


class BenchmarkReport:

    def __init__(self):
        self.records = []

    def add_result(
        self,
        model_name,
        stock_id,
        metrics: dict,
    ):
        row = {
            "model": model_name,
            "stock_id": stock_id,
        }

        row.update(metrics)

        self.records.append(row)

    def to_dataframe(self):

        if not self.records:
            return pd.DataFrame()

        return pd.DataFrame(self.records)

    def save_csv(self, filename="benchmark_result.csv"):

        df = self.to_dataframe()

        output_path = OUTPUT_DIR / filename

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        return output_path

    def ranking(self):

        df = self.to_dataframe()

        if df.empty:
            return df

        return df.sort_values(
            by="accuracy",
            ascending=False
        ).reset_index(drop=True)

    def print_summary(self):

        ranking = self.ranking()

        if ranking.empty:
            print("沒有 Benchmark 結果")
            return

        print("=" * 60)
        print("Benchmark Ranking")
        print("=" * 60)

        print(
            ranking[
                [
                    "model",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                ]
            ]
        )
