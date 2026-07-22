"""
benchmark_runner.py

V11 Benchmark Runner

負責：

Data
 ↓
Models
 ↓
Walk Forward Validation
 ↓
Metrics
 ↓
Summary


不修改正式 predictor.py

"""

import pandas as pd


from predictor import (
    build_feature_data,
    resolve_stock_code
)


from .benchmark_models import (
    create_models
)


from .benchmark_validation import (
    evaluate_model_walk_forward
)


from .benchmark_config import (
    CONFIG
)

from .benchmark_feature_sets import (
    FEATURE_SETS,
    get_feature_set
)



##########################################################################
# Dataset Preparation
##########################################################################

def prepare_benchmark_dataset(
     stock_code,

    feature_columns
):

    """
    建立 Benchmark Dataset

    使用正式 Feature Engineering

    """


    stock_code = resolve_stock_code(
        stock_code
    )


    df = build_feature_data(
        stock_code
    )


    required_columns = [

        "date",

        "Target",

        *feature_columns

    ]



    missing = [

        column

        for column in required_columns

        if column not in df.columns

    ]



    if missing:

        raise ValueError(

            f"Missing columns: {missing}"

        )



    data = df[

        required_columns

    ].copy()



    data["date"] = pd.to_datetime(

        data["date"],

        errors="coerce"

    )



    data = data.dropna()



    data = data.sort_values(

        "date"

    )



    data = data.reset_index(

        drop=True

    )



    X = data[

        feature_columns

    ]



    y = data[

        "Target"

    ]



    dates = data[

        "date"

    ]



    return (

        X,

        y,

        dates,

        data

    )





##########################################################################
# Run Benchmark
##########################################################################

def run_benchmark(

    stock_code,

    feature_name="Baseline"

):

    """

    執行完整 Benchmark

    """



    print("=" * 80)

    print(

        f"V11 Benchmark Start : {stock_code}"

    )

    print("=" * 80)

    feature_columns = get_feature_set(

    feature_name

    )

    X, y, dates, data = prepare_benchmark_dataset(

         stock_code,

        feature_columns

    )



    models = create_models()



    prediction_results = []



    metric_results = []



    for model_name, model in models.items():



        print()

        print(

            f"Testing Model : {model_name}"

        )



        prediction_df, metrics_df = evaluate_model_walk_forward(

            model_name=model_name,

            model=model,

            X=X,

            y=y,

            dates=dates,

            feature_columns=feature_columns,

            n_splits=5

        )



        prediction_results.append(

            prediction_df

        )



        metric_results.append(

            metrics_df

        )





    predictions = pd.concat(

        prediction_results,

        ignore_index=True

    )



    metrics = pd.concat(

        metric_results,

        ignore_index=True

    )



    summary = build_summary(

        metrics

    )



    return {


        "stock":

            stock_code,
               
        "feature_name":

            feature_name,


        "data":

            data,


        "predictions":

            predictions,


        "metrics":

            metrics,


        "summary":

            summary,

     
    }





##########################################################################
# Summary
##########################################################################

def build_summary(

    metrics_df

):

    """

    建立模型比較表

    """



    summary = (

        metrics_df

        .groupby(

            "model"

        )

        [

            [

                "accuracy",

                "precision",

                "recall",

                "f1",

                "roc_auc",

                "confidence"

            ]

        ]

        .mean()

        .reset_index()

    )



    summary["score"] = (

        summary["f1"] * 0.5

        +

        summary["accuracy"] * 0.3

        +

        summary["roc_auc"] * 0.2

    )



    summary = summary.sort_values(

        "score",

        ascending=False

    )



    summary = summary.reset_index(

        drop=True

    )



    summary.insert(

        0,

        "rank",

        range(

            1,

            len(summary)+1

        )

    )



    return summary