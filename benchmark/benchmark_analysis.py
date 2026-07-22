"""
benchmark_analysis.py

V11 Benchmark Analysis

負責：

- Error Analysis
- Prediction Distribution
- Confidence Distribution
- Fold Stability
- Model Comparison

"""

import numpy as np
import pandas as pd



##########################################################################
# Error Analysis
##########################################################################

def analyze_prediction_errors(

    prediction_df

):

    """
    分析錯誤預測

    False Positive:
        預測上漲，實際下跌


    False Negative:
        預測下跌，實際上漲

    """

    df = prediction_df.copy()



    wrong = df[

        df["correct"] == 0

    ].copy()



    if wrong.empty:

        return pd.DataFrame()



    wrong["error_type"] = np.where(

        (

            wrong["actual"] == 1

        )

        &

        (

            wrong["prediction"] == 0

        ),

        "False Negative",

        "False Positive"

    )



    wrong["confidence"] = np.maximum(

        wrong["probability"],

        1 - wrong["probability"]

    )



    wrong = wrong.sort_values(

        "confidence",

        ascending=False

    )



    return wrong





##########################################################################
# Prediction Distribution
##########################################################################

def prediction_distribution(

    prediction_df

):

    """
    預測方向分布

    """

    df = prediction_df.copy()



    total = len(df)



    if total == 0:

        return {}



    predict_up = int(

        (

            df["prediction"] == 1

        ).sum()

    )



    predict_down = int(

        (

            df["prediction"] == 0

        ).sum()

    )



    actual_up = int(

        (

            df["actual"] == 1

        ).sum()

    )



    actual_down = int(

        (

            df["actual"] == 0

        ).sum()

    )



    return {


        "total":

            total,


        "predict_up":

            predict_up,


        "predict_down":

            predict_down,


        "actual_up":

            actual_up,


        "actual_down":

            actual_down,


        "prediction_up_ratio":

            predict_up / total,


        "actual_up_ratio":

            actual_up / total

    }





##########################################################################
# Confidence Distribution
##########################################################################

def confidence_distribution(

    prediction_df

):

    """
    信心值分布分析

    """

    df = prediction_df.copy()



    df["confidence"] = np.maximum(

        df["probability"],

        1 - df["probability"]

    )



    bins = [

        0.50,

        0.60,

        0.70,

        0.80,

        0.90,

        1.00

    ]



    df["confidence_group"] = pd.cut(

        df["confidence"],

        bins=bins,

        include_lowest=True

    )



    results = []



    for name, group in df.groupby(

        "confidence_group",

        observed=False

    ):


        if len(group) == 0:

            continue



        results.append(

            {

                "confidence_range":

                    str(name),


                "samples":

                    len(group),


                "accuracy":

                    group["correct"].mean(),


                "average_probability":

                    group["probability"].mean()

            }

        )



    return pd.DataFrame(results)





##########################################################################
# Fold Stability
##########################################################################

def fold_stability_analysis(

    metrics_df

):

    """
    分析 Walk Forward 穩定性

    """

    columns = [

        "accuracy",

        "precision",

        "recall",

        "f1",

        "roc_auc",

        "confidence"

    ]



    result = []



    for column in columns:


        if column not in metrics_df.columns:

            continue



        result.append(

            {

                "metric":

                    column,


                "mean":

                    metrics_df[column].mean(),


                "std":

                    metrics_df[column].std(),


                "min":

                    metrics_df[column].min(),


                "max":

                    metrics_df[column].max()

            }

        )



    return pd.DataFrame(result)





##########################################################################
# Model Comparison
##########################################################################

def compare_models(

    summary_df

):

    """
    模型排名比較

    """

    df = summary_df.copy()



    df["overall_score"] = (

        df["accuracy"] * 0.3

        +

        df["f1"] * 0.4

        +

        df["roc_auc"] * 0.3

    )



    df = df.sort_values(

        "overall_score",

        ascending=False

    )



    df = df.reset_index(

        drop=True

    )



    # 防止重複 rank 欄位

    if "rank" in df.columns:

        df = df.drop(

            columns=["rank"]

        )



    df.insert(

        0,

        "rank",

        range(

            1,

            len(df)+1

        )

    )



    return df