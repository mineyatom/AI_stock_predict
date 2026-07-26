"""
V11.4 Confidence Threshold Analysis

分析：

不同 Confidence 門檻
對模型準確率影響

"""

from pathlib import Path

import pandas as pd


from sklearn.metrics import accuracy_score, f1_score



def analyze_confidence(

    csv_path

):


    df = pd.read_csv(

        csv_path

    )


    # Confidence

    df["confidence"] = (

        df["raw_probability"]

        .apply(

            lambda x:

            max(x, 1-x)

        )

    )



    results = []



    thresholds = [

        0.50,

        0.55,

        0.60,

        0.65,

        0.70,

        0.75,

        0.80,

        0.85,

    ]



    for threshold in thresholds:


        filtered = df[

            df["confidence"] >= threshold

        ]



        if len(filtered) == 0:

            continue



        prediction = (

            filtered["raw_probability"]

            >= 0.5

        ).astype(int)



        accuracy = accuracy_score(

            filtered["actual"],

            prediction

        )


        f1 = f1_score(

            filtered["actual"],

            prediction,

            zero_division=0

        )


        results.append(

            {

                "threshold":

                    threshold,


                "samples":

                    len(filtered),


                "accuracy":

                    accuracy,


                "f1":

                    f1

            }

        )



    result_df = pd.DataFrame(

        results

    )



    output = (

        Path(__file__).resolve().parent

        /

        "results"

        /

        "confidence_threshold_analysis.csv"

    )



    result_df.to_csv(

        output,

        index=False,

        encoding="utf-8-sig"

    )


    print()

    print("="*80)

    print("Confidence Threshold Analysis")

    print("="*80)


    print(

        result_df.to_string(

            index=False

        )

    )


    print()

    print(

        "Saved:",

        output

    )





if __name__ == "__main__":


    analyze_confidence(

        "benchmark/results/2330_calibration_result.csv"

    )