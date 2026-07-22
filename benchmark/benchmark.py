"""
benchmark.py

V11.1 Feature Benchmark Main Entry

用途：

驗證不同 Feature Set 對模型效果影響。

流程：

Feature Set
    ↓
Model Benchmark
    ↓
Walk Forward Validation
    ↓
Metrics
    ↓
Feature Comparison
    ↓
Report Export

"""


import argparse


import pandas as pd



from .benchmark_runner import (

    run_benchmark

)


from .benchmark_feature_sets import (

    list_feature_sets

)





##########################################################################
# Argument
##########################################################################

def parse_args():

    parser = argparse.ArgumentParser(

        description=

        "AI Stock Predictor V11.1 Feature Benchmark"

    )


    parser.add_argument(

        "stock",

        nargs="?",

        default="2330",

        help="Stock Code"

    )


    return parser.parse_args()





##########################################################################
# Feature Ranking
##########################################################################

def build_feature_ranking(

    results

):

    """

    建立 Feature Set 排名

    """

    rows = []


    for result in results:


        summary = result["summary"]


        best = summary.iloc[0]


        rows.append(

            {

                "feature_set":

                    result["feature_name"],


                "best_model":

                    best["model"],


                "accuracy":

                    best["accuracy"],


                "f1":

                    best["f1"],


                "roc_auc":

                    best["roc_auc"],


                "score":

                    best["score"]

            }

        )



    df = pd.DataFrame(

        rows

    )



    df = df.sort_values(

        "score",

        ascending=False

    )



    df = df.reset_index(

        drop=True

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





##########################################################################
# Main
##########################################################################

def main():


    args = parse_args()


    stock_code = args.stock



    print("=" * 80)

    print(

        "AI Stock Predictor V11.1 Feature Benchmark"

    )

    print("=" * 80)



    print(

        f"Stock : {stock_code}"

    )





    ######################################################################
    # Run Feature Sets
    ######################################################################


    feature_results = []



    for feature_name in list_feature_sets():


        print()

        print("=" * 80)

        print(

            f"Feature Set : {feature_name}"

        )

        print("=" * 80)



        result = run_benchmark(

            stock_code,

            feature_name

        )



        feature_results.append(

            result

        )





        print()

        print(

            result["summary"].to_string(

                index=False

            )

        )





    ######################################################################
    # Feature Comparison
    ######################################################################


    ranking = build_feature_ranking(

        feature_results

    )



    print()

    print("=" * 80)

    print(

        "Feature Set Ranking"

    )

    print("=" * 80)



    print(

        ranking.to_string(

            index=False

        )

    )





    ######################################################################
    # Export

    ######################################################################


    output_path = (

        "benchmark_feature_ranking.csv"

    )


    ranking.to_csv(

        output_path,

        index=False,

        encoding="utf-8-sig"

    )



    print()

    print(

        f"Saved : {output_path}"

    )



    print()

    print("=" * 80)

    print(

        "V11.1 Feature Benchmark Finished"

    )

    print("=" * 80)





if __name__ == "__main__":

    main()