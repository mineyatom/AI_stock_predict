"""
benchmark_config.py

V11 Benchmark Configuration

獨立於 predictor.py 的 Benchmark 設定檔

用途：

- 控制 Benchmark 參數
- 管理模型設定
- 管理 Feature Columns
- 管理輸出路徑

"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List



@dataclass(frozen=True)
class BenchmarkConfig:


    ######################################################################
    # Dataset
    ######################################################################

    # 訓練資料起始日

    train_start_date: str = "2020-01-01"



    ######################################################################
    # Validation
    ######################################################################

    # Walk Forward 最小訓練筆數

    min_train_size: int = 120


    # Walk Forward 分割數

    walk_forward_splits: int = 5



    ######################################################################
    # Prediction
    ######################################################################

    # 預測門檻

    probability_threshold: float = 0.50



    ######################################################################
    # System
    ######################################################################

    # Random Seed

    random_state: int = 42


    # 是否輸出詳細 Log

    verbose: bool = True



    ######################################################################
    # Models
    ######################################################################

    # 啟用模型

    enabled_models: List[str] = field(

        default_factory=lambda: [

            "xgboost",

            "random_forest",

            "logistic_regression",

        ]

    )



    ######################################################################
    # Feature Columns
    ######################################################################

    # 與 predictor.py 保持一致

    feature_columns: List[str] = field(

        default_factory=lambda: [

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

    )



    ######################################################################
    # Output
    ######################################################################

    # Benchmark輸出資料夾

    result_folder: Path = (

        Path(__file__).resolve().parent

        /

        "results"

    )





# Global Config Instance

CONFIG = BenchmarkConfig()