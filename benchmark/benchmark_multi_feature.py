"""
V11.2 Multi Stock Feature Validation

比較 Baseline vs Reduced Feature

"""

import pandas as pd


from .benchmark_runner import run_benchmark



STOCKS = [

    "2330",

    "2454",

    "2317",

    "0050",

    "2308",

    "2382",

    "3661",

]



FEATURES = [

    "Baseline",

    "Reduced",

]





def main():


    results = []


    for stock in STOCKS:


        for feature in FEATURES:


            print()

            print("=" * 80)

            print(
                f"{stock} - {feature}"
            )

            print("=" * 80)



            result = run_benchmark(

                stock,

                feature

            )


            summary = result["summary"]


            best = summary.iloc[0]


            results.append(

                {

                    "stock":

                        stock,


                    "feature":

                        feature,


                    "best_model":

                        best["model"],


                    "accuracy":

                        best["accuracy"],


                    "f1":

                        best["f1"],


                    "roc_auc":

                        best["roc_auc"]

                }

            )



    df = pd.DataFrame(results)



    df.to_csv(

        "multi_feature_comparison.csv",

        index=False,

        encoding="utf-8-sig"

    )



    print()

    print("=" * 80)

    print("V11.2 Finished")

    print("=" * 80)


    print(df.to_string(index=False))





if __name__ == "__main__":

    main()