"""
benchmark_metrics.py

V11 Benchmark Metrics

負責：

- Accuracy
- Precision
- Recall
- F1
- ROC AUC
- Confidence

"""

import numpy as np


from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    confusion_matrix

)





##########################################################################
# Calculate Metrics
##########################################################################

def calculate_metrics(

    actual,

    prediction,

    probability

):

    """
    計算模型評估指標


    Parameters:

    actual:
        真實結果


    prediction:
        模型預測結果


    probability:
        預測上漲機率



    Returns:

        dict

    """



    metrics = {}



    ######################################################################
    # Basic Metrics
    ######################################################################


    metrics["accuracy"] = accuracy_score(

        actual,

        prediction

    )



    metrics["precision"] = precision_score(

        actual,

        prediction,

        zero_division=0

    )



    metrics["recall"] = recall_score(

        actual,

        prediction,

        zero_division=0

    )



    metrics["f1"] = f1_score(

        actual,

        prediction,

        zero_division=0

    )




    ######################################################################
    # ROC AUC
    ######################################################################


    try:

        metrics["roc_auc"] = roc_auc_score(

            actual,

            probability

        )


    except ValueError:

        metrics["roc_auc"] = 0





    ######################################################################
    # Confidence
    ######################################################################


    confidence = np.maximum(

        probability,

        1 - probability

    )



    metrics["confidence"] = np.mean(

        confidence

    )





    ######################################################################
    # Confusion Matrix
    ######################################################################


    tn, fp, fn, tp = confusion_matrix(

        actual,

        prediction,

        labels=[0,1]

    ).ravel()



    metrics["true_negative"] = int(tn)


    metrics["false_positive"] = int(fp)


    metrics["false_negative"] = int(fn)


    metrics["true_positive"] = int(tp)




    return metrics





##########################################################################
# Batch Metrics Summary
##########################################################################

def summarize_metrics(

    metrics_df

):

    """
    多 Fold 統計

    """



    columns = [

        "accuracy",

        "precision",

        "recall",

        "f1",

        "roc_auc",

        "confidence"

    ]



    summary = {}



    for column in columns:


        if column in metrics_df.columns:


            summary[column] = {

                "mean":

                    metrics_df[column].mean(),


                "std":

                    metrics_df[column].std(),


                "min":

                    metrics_df[column].min(),


                "max":

                    metrics_df[column].max()

            }



    return summary