from sqlalchemy.exc import IntegrityError

from database import SessionLocal
from models import Prediction

import pandas as pd



def normalize_date(date_value: str) -> str:
    """
    統一日期格式為 YYYY-MM-DD。
    """

    return pd.to_datetime(
        date_value
    ).strftime("%Y-%m-%d")


def create_prediction(
    predict_date: str,
    stock_code: str,
    stock_name: str,
    prediction_text: str,
    confidence: float,
    up_probability: float,
    down_probability: float,
    predict_close: float,
    lower_price: float,
    upper_price: float,
) -> bool:
    """
    新增一筆預測資料。

    回傳：
    True  = 新增成功
    False = 已存在或新增失敗
    """


    predict_date = normalize_date(
    predict_date
    )

    db = SessionLocal()

    try:
        # ==========================
        # 檢查是否已存在
        # ==========================
        existing = (
            db.query(Prediction)
            .filter(
                Prediction.predict_date == predict_date,
                Prediction.stock_code == stock_code,
            )
            .first()
        )

        if existing:
            print(
                f"⏭️ SQLite 已存在："
                f"{predict_date} {stock_code}"
            )
            return False

        # ==========================
        # 建立預測資料
        # ==========================
        prediction = Prediction(
            predict_date=predict_date,
            stock_code=stock_code,
            stock_name=stock_name,
            prediction_text=prediction_text,
            confidence=float(confidence),
            up_probability=float(up_probability),
            down_probability=float(down_probability),
            predict_close=float(predict_close),
            lower_price=float(lower_price),
            upper_price=float(upper_price),
        )

        db.add(prediction)
        db.commit()

        print(
            f"💾 SQLite 預測已儲存："
            f"{predict_date} {stock_code}"
        )

        return True

    except IntegrityError:
        db.rollback()

        print(
            f"⏭️ SQLite 重複資料："
            f"{predict_date} {stock_code}"
        )

        return False

    except Exception as e:
        db.rollback()

        print(
            f"❌ SQLite 儲存失敗："
            f"{predict_date} {stock_code} "
            f"原因：{e}"
        )

        return False

    finally:
        db.close()





def prediction_exists(
    predict_date: str,
    stock_code: str | None = None,
) -> bool:
    """
    檢查預測資料是否存在。

    stock_code 有提供：
    → 檢查指定日期 + 指定股票

    stock_code 沒提供：
    → 檢查指定日期是否有任何預測
    """

    predict_date = normalize_date(
    predict_date
    )

    db = SessionLocal()

    try:
        query = (
            db.query(Prediction)
            .filter(
                Prediction.predict_date == predict_date
            )
        )

        if stock_code is not None:
            query = query.filter(
                Prediction.stock_code == stock_code
            )

        return query.first() is not None

    finally:
        db.close()


def update_prediction_validation(
    predict_date: str,
    stock_code: str,
    actual_close: float,
    actual_change: str,
    is_correct: str,
) -> bool:
    """
    更新一筆預測的實際驗證結果。
    """

    db = SessionLocal()

    try:
        prediction = (
            db.query(Prediction)
            .filter(
                Prediction.predict_date == predict_date,
                Prediction.stock_code == stock_code,
            )
            .first()
        )

        if prediction is None:
            print(
                f"⚠️ SQLite 找不到預測資料："
                f"{predict_date} {stock_code}"
            )
            return False

        prediction.actual_close = float(actual_close)
        prediction.actual_change = actual_change
        prediction.is_correct = is_correct

        db.commit()

        print(
            f"✅ SQLite 驗證已更新："
            f"{predict_date} {stock_code}"
        )

        return True

    except Exception as e:
        db.rollback()

        print(
            f"❌ SQLite 驗證更新失敗："
            f"{predict_date} {stock_code} "
            f"原因：{e}"
        )

        return False

    finally:
        db.close()

def update_prediction_date(
    old_predict_date: str,
    new_predict_date: str,
    stock_code: str,
) -> bool:
    """
    更新預測日期。

    用於颱風假、臨時休市等情況，
    將原本的預測日期順延到下一個交易日。
    """

    old_predict_date = normalize_date(
        old_predict_date
    )

    new_predict_date = normalize_date(
        new_predict_date
    )

    db = SessionLocal()

    try:
        prediction = (
            db.query(Prediction)
            .filter(
                Prediction.predict_date == old_predict_date,
                Prediction.stock_code == stock_code,
            )
            .first()
        )

        if prediction is None:
            print(
                f"⚠️ SQLite 找不到預測資料："
                f"{old_predict_date} {stock_code}"
            )
            return False

        # 檢查新日期是否已經有相同股票
        existing = (
            db.query(Prediction)
            .filter(
                Prediction.predict_date == new_predict_date,
                Prediction.stock_code == stock_code,
            )
            .first()
        )

        if existing is not None:
            print(
                f"⚠️ SQLite 新日期已有預測："
                f"{new_predict_date} {stock_code}"
            )
            return False

        prediction.predict_date = new_predict_date

        db.commit()

        print(
            f"📅 SQLite 預測日期已順延："
            f"{stock_code} "
            f"{old_predict_date} → {new_predict_date}"
        )

        return True

    except Exception as e:
        db.rollback()

        print(
            f"❌ SQLite 日期更新失敗："
            f"{old_predict_date} → {new_predict_date} "
            f"{stock_code} "
            f"原因：{e}"
        )

        return False

    finally:
        db.close()        