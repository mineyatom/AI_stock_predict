import pandas as pd

from database import SessionLocal
from models import Prediction


def normalize_db_dates():
    """
    將 predictions 資料表中的 predict_date
    統一轉成 YYYY-MM-DD。
    """

    print("🔄 開始統一 SQLite 日期格式")

    db = SessionLocal()

    updated_count = 0
    skipped_count = 0
    failed_count = 0

    try:
        predictions = db.query(Prediction).all()

        for prediction in predictions:
            old_date = str(prediction.predict_date).strip()

            try:
                new_date = pd.to_datetime(
                    old_date
                ).strftime("%Y-%m-%d")

            except Exception as e:
                print(
                    f"❌ 日期轉換失敗："
                    f"id={prediction.id} "
                    f"date={old_date} "
                    f"原因：{e}"
                )

                failed_count += 1
                continue

            if old_date == new_date:
                skipped_count += 1
                continue

            # 防止轉換後出現同日期、同股票的重複資料
            duplicate = (
                db.query(Prediction)
                .filter(
                    Prediction.predict_date == new_date,
                    Prediction.stock_code == prediction.stock_code,
                    Prediction.id != prediction.id,
                )
                .first()
            )

            if duplicate is not None:
                print(
                    f"⚠️ 轉換後會重複，略過："
                    f"{old_date} → {new_date} "
                    f"{prediction.stock_code}"
                )

                skipped_count += 1
                continue

            prediction.predict_date = new_date
            updated_count += 1

            print(
                f"✅ 日期已更新："
                f"{old_date} → {new_date} "
                f"{prediction.stock_code}"
            )

        db.commit()

    except Exception as e:
        db.rollback()

        print(
            f"❌ SQLite 日期統一失敗：{e}"
        )

        return

    finally:
        db.close()

    print()
    print("=" * 50)
    print("SQLite 日期格式統一完成")
    print("=" * 50)
    print(f"更新：{updated_count}")
    print(f"略過：{skipped_count}")
    print(f"失敗：{failed_count}")


if __name__ == "__main__":
    normalize_db_dates()