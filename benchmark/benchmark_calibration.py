"""
benchmark_calibration.py

V11 Probability Calibration Analysis

分析：

模型輸出的 probability

是否代表真正可信度。

例如：

模型說：

90%

是否真的有約90%正確率。

"""

import numpy as np
import pandas as pd



DEFAULT_BINS = [

    0.50,

    0.55,

    0.60,

    0.65,

    0.70,

    0.75,

    0.80,

    0.85,

    0.90,

    0.95,

    1.00,

]



def create_probability_bins(
    prediction_df,
    bins=None
):
    """
    建立 Confidence 分桶
    """

    if bins is None:

        bins = DEFAULT_BINS


    df = prediction_df.copy()


    df["confidence"] = np.maximum(

        df["probability"],

        1 - df["probability"]

    )


    df["confidence_bin"] = pd.cut(

        df["confidence"],

        bins=bins,

        include_lowest=True,

    )


    return df





def calculate_calibration_table(

    prediction_df,

    bins=None

):
    """
    建立 Calibration Table


    欄位：

    confidence range

    sample count

    average confidence

    actual accuracy

    gap

    """


    df = create_probability_bins(

        prediction_df,

        bins

    )


    groups = []


    for name, group in df.groupby(

        "confidence_bin",

        observed=False

    ):


        if len(group) == 0:

            continue



        avg_confidence = group[

            "confidence"

        ].mean()



        accuracy = group[

            "correct"

        ].mean()



        groups.append(

            {

                "confidence_range":

                    str(name),


                "samples":

                    len(group),


                "average_confidence":

                    avg_confidence,


                "actual_accuracy":

                    accuracy,


                "calibration_gap":

                    abs(

                        avg_confidence

                        -

                        accuracy

                    )

            }

        )


    result = pd.DataFrame(

        groups

    )


    return result





def calculate_expected_calibration_error(

    calibration_df

):
    """
    ECE

    Expected Calibration Error

    """

    if calibration_df.empty:

        return 0



    total_samples = calibration_df[

        "samples"

    ].sum()



    ece = 0



    for _, row in calibration_df.iterrows():


        weight = (

            row["samples"]

            /

            total_samples

        )


        ece += (

            weight

            *

            row["calibration_gap"]

        )



    return float(ece)





def calibration_summary(

    calibration_df

):
    """
    文字摘要
    """


    if calibration_df.empty:

        return {

            "status":

                "No Data"

        }



    ece = calculate_expected_calibration_error(

        calibration_df

    )



    if ece < 0.05:

        status = "Excellent Calibration"


    elif ece < 0.10:

        status = "Acceptable Calibration"


    else:

        status = "Need Calibration"



    return {

        "ece":

            round(ece,4),


        "status":

            status

    }





def confidence_accuracy_analysis(

    prediction_df

):
    """
    分析高信心預測是否比較準

    """

    df = prediction_df.copy()



    df["confidence"] = np.maximum(

        df["probability"],

        1 - df["probability"]

    )


    levels = [

        0.50,

        0.60,

        0.70,

        0.80,

        0.90,

    ]



    results = []



    for level in levels:


        selected = df[

            df["confidence"] >= level

        ]


        if len(selected) == 0:

            continue



        results.append(

            {

                "confidence_level":

                    level,


                "samples":

                    len(selected),


                "accuracy":

                    selected["correct"].mean()

            }

        )



    return pd.DataFrame(

        results

    )