"""
benchmark_export.py

V11 Benchmark Export Module

負責：

- CSV Export
- JSON Export
- HTML Report
- Markdown Report

"""

from pathlib import Path
from datetime import datetime
import json

import pandas as pd


from .benchmark_config import (
    CONFIG
)



##########################################################################
# Folder
##########################################################################

def create_output_folder():

    folders = [

        CONFIG.result_folder,

        CONFIG.result_folder / "csv",

        CONFIG.result_folder / "json",

        CONFIG.result_folder / "html",

        CONFIG.result_folder / "markdown",

    ]


    for folder in folders:

        folder.mkdir(

            parents=True,

            exist_ok=True

        )


##########################################################################
# CSV Export
##########################################################################

def export_csv(

    stock_code,

    summary_df,

    metrics_df,

    prediction_df,

):

    create_output_folder()


    summary_path = (

        CONFIG.result_folder

        /

        "csv"

        /

        f"{stock_code}_summary.csv"

    )


    metrics_path = (

        CONFIG.result_folder

        /

        "csv"

        /

        f"{stock_code}_metrics.csv"

    )


    prediction_path = (

        CONFIG.result_folder

        /

        "csv"

        /

        f"{stock_code}_predictions.csv"

    )



    summary_df.to_csv(

        summary_path,

        index=False,

        encoding="utf-8-sig"

    )


    metrics_df.to_csv(

        metrics_path,

        index=False,

        encoding="utf-8-sig"

    )


    prediction_df.to_csv(

        prediction_path,

        index=False,

        encoding="utf-8-sig"

    )


    return {

        "summary":

            summary_path,

        "metrics":

            metrics_path,

        "prediction":

            prediction_path

    }




##########################################################################
# JSON Export
##########################################################################

def export_json(

    stock_code,

    data,

):

    create_output_folder()


    path = (

        CONFIG.result_folder

        /

        "json"

        /

        f"{stock_code}_report.json"

    )


    def convert(obj):

        if hasattr(

            obj,

            "item"

        ):

            return obj.item()


        if isinstance(

            obj,

            pd.DataFrame

        ):

            return obj.to_dict(

                orient="records"

            )


        return str(obj)



    with open(

        path,

        "w",

        encoding="utf-8"

    ) as file:


        json.dump(

            data,

            file,

            ensure_ascii=False,

            indent=4,

            default=convert

        )


    return path




##########################################################################
# HTML Export
##########################################################################

def export_html(

    stock_code,

    summary_df,

    metrics_df,

):

    create_output_folder()


    path = (

        CONFIG.result_folder

        /

        "html"

        /

        f"{stock_code}_benchmark.html"

    )



    html = f"""

<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<title>
V11 Benchmark Report
</title>


<style>

body {{

font-family:
Arial;

padding:
30px;

background:
#f5f5f5;

}}


table {{

border-collapse:
collapse;

width:
100%;

background:
white;

}}


th {{

background:
#34495e;

color:
white;

padding:
10px;

}}


td {{

padding:
8px;

border:
1px solid #ddd;

text-align:
center;

}}


</style>


</head>



<body>


<h1>
AI Stock Predictor V11 Benchmark
</h1>


<h2>
Stock :
{stock_code}
</h2>


<p>

Generated :

{datetime.now()}

</p>



<h2>
Model Ranking
</h2>


{summary_df.to_html(

index=False

)}



<h2>
Fold Metrics
</h2>


{metrics_df.to_html(

index=False

)}


</body>


</html>

"""


    with open(

        path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            html

        )


    return path




##########################################################################
# Markdown Export
##########################################################################

def export_markdown(

    stock_code,

    summary_df,

):

    create_output_folder()


    path = (

        CONFIG.result_folder

        /

        "markdown"

        /

        f"{stock_code}_report.md"

    )


    best = summary_df.iloc[0]



    content = f"""

# AI Stock Predictor V11 Benchmark Report


## Stock

{stock_code}



## Best Model


Model:

{best['model']}



Accuracy:

{best['accuracy']:.4f}



F1:

{best['f1']:.4f}



ROC AUC:

{best['roc_auc']:.4f}



## Ranking



{summary_df.to_markdown(

index=False

)}



Generated:

{datetime.now()}

"""


    with open(

        path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            content

        )


    return path