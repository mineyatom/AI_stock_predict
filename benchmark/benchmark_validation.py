"""
benchmark_validation.py

V11 Benchmark Validation Engine

負責：
- Time Series Split
- Walk Forward Validation
- Model Evaluation Pipeline

不修改正式 predictor.py
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from .benchmark_metrics import (
    calculate_metrics
)


def create_walk_forward_split(
    data_length,
    n_splits=5
):
    """
    建立時間序列 Walk Forward

    避免未來資料洩漏
    """

    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    splits = []

    for train_idx, test_idx in splitter.split(
        np.arange(data_length)
    ):

        splits.append(
            (
                train_idx,
                test_idx
            )
        )

    return splits



def evaluate_model_walk_forward(
    model_name,
    model,
    X,
    y,
    dates,
    feature_columns,
    n_splits=5
):
    """
    單模型 Walk Forward 評估
    """

    splits = create_walk_forward_split(
        len(X),
        n_splits
    )


    prediction_records = []

    fold_results = []


    fold_number = 1


    for train_idx, test_idx in splits:


        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]


        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]


        model.fit(
            X_train,
            y_train
        )


        prediction = model.predict(
            X_test
        )


        probability = model.predict_proba(
            X_test
        )[:,1]


        metrics = calculate_metrics(
            y_test,
            prediction,
            probability
        )


        metrics["model"] = model_name

        metrics["fold"] = fold_number


        fold_results.append(
            metrics
        )


        fold_df = pd.DataFrame(
            {

                "date":
                    dates.iloc[test_idx].values,


                "model":
                    model_name,


                "fold":
                    fold_number,


                "actual":
                    y_test.values,


                "prediction":
                    prediction,


                "probability":
                    probability,


                "confidence":
                    np.maximum(
                        probability,
                        1 - probability
                    ),


                "correct":
                    (
                        prediction
                        ==
                        y_test.values
                    ).astype(int)

            }
        )


        prediction_records.append(
            fold_df
        )


        fold_number += 1


    prediction_df = pd.concat(
        prediction_records,
        ignore_index=True
    )


    metrics_df = pd.DataFrame(
        fold_results
    )


    return (
        prediction_df,
        metrics_df
    )