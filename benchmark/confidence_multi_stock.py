"""
V11.6 Confidence Multi Stock Validation

確認：
高 Confidence 是否在多股票中有效

"""


from pathlib import Path

import pandas as pd



from .calibration_test import test_calibration





STOCKS = [

    "2330",

    "2454",

    "2317",

    "0050",

    "2308",

    "2382",

    "3661",

]





def calculate_confidence(

    csv_path,

    stock

):


    df = pd.read_csv(

        csv_path

    )



    df["confidence"] = (

        df["raw_probability"]

        .apply(

            lambda x:

            max(x, 1-x)

        )

    )



    results = []



    for threshold in [

        0.5,

        0.6,

        0.7,

        0.8

    ]:


        temp = df[

            df["confidence"] >= threshold

        ]



        if len(temp) == 0:

            continue



        prediction = (

            temp["raw_probability"]

            >= 0.5

        ).astype(int)



        accuracy = (

            prediction

            == temp["actual"]

        ).mean()



        results.append(

            {

                "stock":

                    stock,


                "threshold":

                    threshold,


                "samples":

                    len(temp),


                "accuracy":

                    accuracy

            }

        )


    return results





def main():


    all_results = []



    output_folder = (

        Path(__file__).resolve().parent

        /

        "results"

    )



    output_folder.mkdir(

        exist_ok=True

    )



    for stock in STOCKS:


        print()

        print("="*80)

        print(

            f"Testing {stock}"

        )

        print("="*80)



        # 產生 calibration csv

        test_calibration(

            stock

        )



        csv_path = (

            output_folder

            /

            f"{stock}_calibration_result.csv"

        )


        result = calculate_confidence(

            csv_path,

            stock

        )


        all_results.extend(

            result

        )



    df = pd.DataFrame(

        all_results

    )



    output = (

        output_folder

        /

        "confidence_multi_stock.csv"

    )



    df.to_csv(

        output,

        index=False,

        encoding="utf-8-sig"

    )



    print()

    print("="*80)

    print("Finished")

    print("="*80)

    print(

        df.to_string(

            index=False

        )

    )





if __name__ == "__main__":

    main()