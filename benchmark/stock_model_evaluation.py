"""
V11.7 Stock Model Evaluation

整合：

Feature Benchmark
Model Benchmark
Confidence Validation

產生股票個別建議

"""


from pathlib import Path

import pandas as pd





RESULT_FOLDER = (

    Path(__file__).resolve().parent

    /

    "results"

)





def load_csv(filename):

    return pd.read_csv(

        RESULT_FOLDER

        /

        filename

    )





def evaluate():


    feature_df = load_csv(

        "multi_feature_comparison.csv"

    )


    confidence_df = load_csv(

        "confidence_multi_stock.csv"

    )



    results = []



    stocks = feature_df["stock"].unique()



    for stock in stocks:



        stock_feature = feature_df[

            feature_df["stock"] == stock

        ]



        best_feature_row = (

            stock_feature

            .sort_values(

                "accuracy",

                ascending=False

            )

            .iloc[0]

        )




        confidence_stock = confidence_df[

            confidence_df["stock"] == stock

        ]



        base_accuracy = (

            confidence_stock[

                confidence_stock["threshold"] == 0.5

            ]

            ["accuracy"]

            .iloc[0]

        )



        high_conf = confidence_stock[

            confidence_stock["threshold"] == 0.7

        ]



        if len(high_conf) > 0:


            high_accuracy = (

                high_conf["accuracy"]

                .iloc[0]

            )


            confidence_valid = (

                high_accuracy > base_accuracy

            )


        else:


            confidence_valid = False




        results.append(

            {

                "stock":

                    stock,


                "best_feature":

                    best_feature_row["feature"],


                "best_model":

                    best_feature_row["best_model"],


                "accuracy":

                    best_feature_row["accuracy"],


                "confidence_valid":

                    confidence_valid

            }

        )




    result_df = pd.DataFrame(

        results

    )



    output = (

        RESULT_FOLDER

        /

        "stock_model_evaluation.csv"

    )



    result_df.to_csv(

        output,

        index=False,

        encoding="utf-8-sig"

    )



    print()

    print("="*80)

    print("Stock Model Evaluation")

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

    evaluate()