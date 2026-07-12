import os

import pandas as pd
from sqlalchemy.exc import IntegrityError

from database import SessionLocal
from models import Prediction


CSV_FILE = "prediction_log.csv"


def clean_float(value):
    """
    將 CSV 內容安全轉成 float。

    空白、NaN、無效內容：
    → None
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return None

    # 移除百分比符號
    value = value.replace("%", "")

    try:
        return float(value)

    except (ValueError, TypeError):
        return None


def clean_string(value):
    """
    將 CSV 內容安全轉成字串。

    空白、NaN：
    → None
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return None

    return value


def migrate_csv_to_db():

    print("🚀 開始遷移 CSV → SQLite")

    if not os.path.exists(CSV_FILE):
        print(f"❌ 找不到 {CSV_FILE}")
        return

    df = pd.read_csv(
        CSV_FILE,
        encoding="utf-8-sig",
        dtype={
            "股票代號": str
        }
    )

    if df.empty:
        print("❌ CSV 沒有資料")
        return

    db = SessionLocal()

    inserted_count = 0
    skipped_count = 0
    failed_count = 0

    try:

        for index, row in df.iterrows():

            predict_date = clean_string(
                row.get("預測日期")
            )

            stock_code = clean_string(
                row.get("股票代號")
            )

            # 必要欄位不存在，直接跳過
            if not predict_date or not stock_code:

                print(
                    f"⚠️ 第 {index + 2} 列缺少必要資料，跳過"
                )

                failed_count += 1
                continue

            # ==========================
            # 防止重複資料
            # ==========================
            existing = (
                db.query(Prediction)
                .filter(
                    Prediction.predict_date
                    == predict_date,

                    Prediction.stock_code
                    == stock_code
                )
                .first()
            )

            if existing:

                print(
                    f"⏭️ 已存在，跳過："
                    f"{predict_date} "
                    f"{stock_code}"
                )

                skipped_count += 1
                continue

            prediction = Prediction(

                predict_date=predict_date,

                stock_code=stock_code,

                stock_name=(
                    clean_string(
                        row.get("股票名稱")
                    )
                    or stock_code
                ),

                prediction_text=(
                    clean_string(
                        row.get("預測結果")
                    )
                    or "未知"
                ),

                confidence=(
                    clean_float(
                        row.get("信心值")
                    )
                    or 0.0
                ),

                up_probability=(
                    clean_float(
                        row.get("上漲機率")
                    )
                    or 0.0
                ),

                down_probability=(
                    clean_float(
                        row.get("下跌機率")
                    )
                    or 0.0
                ),

                predict_close=(
                    clean_float(
                        row.get("隔日預測參考價")
                    )
                    or 0.0
                ),

                lower_price=(
                    clean_float(
                        row.get("預測區間下緣")
                    )
                    or 0.0
                ),

                upper_price=(
                    clean_float(
                        row.get("預測區間上緣")
                    )
                    or 0.0
                ),

                actual_close=clean_float(
                    row.get("實際收盤價")
                ),

                actual_change=clean_string(
                    row.get("實際漲跌")
                ),

                is_correct=clean_string(
                    row.get("是否預測正確")
                ),
            )

            db.add(prediction)

            try:
                db.commit()

                inserted_count += 1

                print(
                    f"✅ 已匯入："
                    f"{predict_date} "
                    f"{stock_code}"
                )

            except IntegrityError:

                db.rollback()

                skipped_count += 1

                print(
                    f"⏭️ 重複資料，跳過："
                    f"{predict_date} "
                    f"{stock_code}"
                )

            except Exception as e:

                db.rollback()

                failed_count += 1

                print(
                    f"❌ 匯入失敗："
                    f"{predict_date} "
                    f"{stock_code} "
                    f"原因：{e}"
                )

    finally:
        db.close()

    print()
    print("=" * 50)
    print("CSV → SQLite 遷移完成")
    print("=" * 50)
    print(f"新增：{inserted_count}")
    print(f"跳過：{skipped_count}")
    print(f"失敗：{failed_count}")


if __name__ == "__main__":
    migrate_csv_to_db()