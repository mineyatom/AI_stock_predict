"""
calibration_test.py

V11.3 Calibration Test

比較：

Raw Probability
vs
Calibrated Probability

目的：

確認模型信心值是否需要校正

"""


from pathlib import Path


import pandas as pd



from sklearn.calibration import CalibratedClassifierCV

from sklearn.frozen import FrozenEstimator


from sklearn.metrics import (

    accuracy_score,

    roc_auc_score

)



from .benchmark_models import create_models

from .benchmark_runner import prepare_benchmark_dataset

from .benchmark_feature_sets import get_feature_set





##########################################################################
# Calibration Test
##########################################################################

def test_calibration(

    stock="2330"

):


    ######################################################################
    # Feature
    ######################################################################


    feature_columns = get_feature_set(

        "Baseline"

    )



    X, y, dates, data = prepare_benchmark_dataset(

        stock,

        feature_columns

    )




    ######################################################################
    # Model
    ######################################################################


    models = create_models()



    print(

        "Available Models:",

        models.keys()

    )



    model = models["XGBoost"]





    ######################################################################
    # Train/Test Split
    ######################################################################


    split = int(

        len(X) * 0.8

    )



    X_train = X.iloc[:split]

    X_test = X.iloc[split:]



    y_train = y.iloc[:split]

    y_test = y.iloc[split:]





    ######################################################################
    # Raw Model
    ######################################################################


    model.fit(

        X_train,

        y_train

    )



    raw_probability = model.predict_proba(

        X_test

    )[:,1]



    raw_prediction = (

        raw_probability >= 0.5

    ).astype(int)





    ######################################################################
    # Calibration
    ######################################################################


    calibrated_model = CalibratedClassifierCV(

        FrozenEstimator(model),

        method="isotonic"

    )



    calibrated_model.fit(

        X_train,

        y_train

    )



    calibrated_probability = calibrated_model.predict_proba(

        X_test

    )[:,1]



    calibrated_prediction = (

        calibrated_probability >= 0.5

    ).astype(int)





    ######################################################################
    # Result
    ######################################################################


    result = pd.DataFrame({

        "actual":

            y_test.values,


        "raw_probability":

            raw_probability,


        "calibrated_probability":

            calibrated_probability,


        "raw_prediction":

            raw_prediction,


        "calibrated_prediction":

            calibrated_prediction

    })





    ######################################################################
    # Metrics
    ######################################################################


    print()

    print("=" * 80)

    print("Raw Probability")

    print("=" * 80)



    print(

        "Accuracy:",

        accuracy_score(

            y_test,

            raw_prediction

        )

    )



    print(

        "ROC AUC:",

        roc_auc_score(

            y_test,

            raw_probability

        )

    )





    print()

    print("=" * 80)

    print("Calibrated Probability")

    print("=" * 80)



    print(

        "Accuracy:",

        accuracy_score(

            y_test,

            calibrated_prediction

        )

    )



    print(

        "ROC AUC:",

        roc_auc_score(

            y_test,

            calibrated_probability

        )

    )





    ######################################################################
    # Export
    ######################################################################


    output_folder = (

        Path(__file__).resolve().parent

        /

        "results"

    )



    output_folder.mkdir(

        parents=True,

        exist_ok=True

    )



    output_path = (

        output_folder

        /

        f"{stock}_calibration_result.csv"

    )



    result.to_csv(

        output_path,

        index=False,

        encoding="utf-8-sig"

    )



    print()

    print(

        f"Saved : {output_path}"

    )





if __name__ == "__main__":


    test_calibration()