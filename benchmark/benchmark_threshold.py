"""
benchmark_threshold.py

V11 Threshold Evaluation

用途：

分析：

0.50
0.55
0.60
0.65
0.70
0.75

不同信心門檻下：

- Accuracy
- Precision
- Recall
- F1
- Sample Count

協助判斷：

模型信心值是否合理

"""


import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = [

    0.50,

    0.55,

    0.60,

    0.65,

    0.70,

    0.75,

]



def evaluate_thresholds(

    prediction_df,

    thresholds=None

):
    """
    測試不同 threshold

    prediction_df 必須包含：

    actual
    probability

    """


    if thresholds is None:

        thresholds = DEFAULT_THRESHOLDS



    results = []



    for threshold in thresholds:


        df = prediction_df.copy()



        df["threshold_prediction"] = (

            df["probability"]

            >= threshold

        ).astype(int)



        actual = df["actual"]

        pred = df["threshold_prediction"]



        tp = (

            (

                actual == 1

            )

            &

            (

                pred == 1

            )

        ).sum()



        tn = (

            (

                actual == 0

            )

            &

            (

                pred == 0

            )

        ).sum()



        fp = (

            (

                actual == 0

            )

            &

            (

                pred == 1

            )

        ).sum()



        fn = (

            (

                actual == 1

            )

            &

            (

                pred == 0

            )

        ).sum()



        total = len(df)



        accuracy = (

            tp + tn

        ) / total



        precision = (

            tp /

            (tp + fp)

            if (

                tp + fp

            ) > 0

            else 0

        )



        recall = (

            tp /

            (tp + fn)

            if (

                tp + fn

            ) > 0

            else 0

        )



        f1 = (

            2 *

            precision *

            recall

            /

            (

                precision +

                recall

            )

            if (

                precision +

                recall

            ) > 0

            else 0

        )



        results.append(

            {

                "threshold":

                    threshold,


                "samples":

                    total,


                "accuracy":

                    accuracy,


                "precision":

                    precision,


                "recall":

                    recall,


                "f1":

                    f1,


                "tp":

                    int(tp),


                "tn":

                    int(tn),


                "fp":

                    int(fp),


                "fn":

                    int(fn)

            }

        )



    result_df = pd.DataFrame(

        results

    )



    result_df = result_df.sort_values(

        "f1",

        ascending=False

    )



    result_df = result_df.reset_index(

        drop=True

    )



    return result_df





def get_best_threshold(

    threshold_df,

    metric="f1"

):
    """
    找最佳 threshold

    預設以 F1 判斷

    """


    if threshold_df.empty:

        return None



    best = threshold_df.sort_values(

        metric,

        ascending=False

    ).iloc[0]



    return {

        "threshold":

            float(best["threshold"]),


        "metric":

            metric,


        "score":

            float(best[metric])

    }





def threshold_summary(

    threshold_df

):

    """
    產生文字摘要
    """


    if threshold_df.empty:

        return "No threshold result"



    best = threshold_df.iloc[0]



    return (

        f"最佳 Threshold: "

        f"{best['threshold']:.2f}\n"

        f"F1: "

        f"{best['f1']:.4f}\n"

        f"Accuracy: "

        f"{best['accuracy']:.4f}"

    )