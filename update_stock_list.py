import os
from io import StringIO

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()

os.makedirs("data", exist_ok=True)


def get_stock_table(url):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        verify=False
    )

    response.encoding = "big5"

    tables = pd.read_html(
        StringIO(response.text)
    )

    return tables[0]


def parse_stock_list(url, suffix):

    df = get_stock_table(url)

    df.columns = df.iloc[0]
    df = df.iloc[1:]

    df = df[
        df["有價證券代號及名稱"]
        .notna()
    ]

    text = (
        df["有價證券代號及名稱"]
        .astype(str)
    )

    extracted = text.str.extract(
        r"^(\d{4})\s+(.+)$"
    )

    df["stock_code"] = extracted[0]
    df["stock_name"] = extracted[1]

    df = df[
        df["stock_code"].notna()
    ]

    df["stock_code"] = (
        df["stock_code"]
        + suffix
    )

    return df[
        [
            "stock_code",
            "stock_name"
        ]
    ]


# ===== 上市 =====
twse_url = (
    "https://isin.twse.com.tw/"
    "isin/C_public.jsp?strMode=2"
)

# ===== 上櫃 =====
tpex_url = (
    "https://isin.twse.com.tw/"
    "isin/C_public.jsp?strMode=4"
)

twse_df = parse_stock_list(
    twse_url,
    ".TW"
)

tpex_df = parse_stock_list(
    tpex_url,
    ".TWO"
)

stock_df = pd.concat(
    [
        twse_df,
        tpex_df
    ],
    ignore_index=True
)

stock_df = stock_df.drop_duplicates(
    subset=["stock_code"]
)

stock_df.to_csv(
    "data/stock_names.csv",
    index=False,
    encoding="utf-8-sig"
)

print("✅ stock_names.csv 更新完成")
print(
    f"共 {len(stock_df)} 檔股票"
)