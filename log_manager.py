import os
import pandas as pd


def get_prediction_history():

    log_file = (
        "prediction_log.csv"
    )

    if not os.path.exists(
        log_file
    ):
        return []

    df = pd.read_csv(
        log_file,
        encoding="utf-8-sig"
    )

    # ===== 美化實際收盤價 =====
    df["實際收盤價"] = (
        pd.to_numeric(
            df["實際收盤價"],
            errors="coerce"
        )
        .apply(
            lambda x:
            int(x)
            if pd.notna(x)
            and float(x).is_integer()
            else x
        )
    )

    df = df.fillna("")

    history = df.to_dict(
        orient="records"
    )

    validated_df = df[
        df["是否預測正確"] != ""
    ]

    validated_count = len(
        validated_df
    )

    total_count = len(df)

    if validated_count > 0:

        accuracy = round(
            (
                validated_df[
                    "是否預測正確"
                ]
                == "正確"
            ).mean() * 100,
            2
        )

    else:
        accuracy = 0

    return {
        "history": history,
        "accuracy": accuracy,
        "validated_count": validated_count,
        "total_count": total_count
    }