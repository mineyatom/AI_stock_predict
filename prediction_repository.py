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
            confidence=round(float(confidence), 2),
            up_probability=round(float(up_probability), 2),
            down_probability=round(float(down_probability), 2),
            predict_close=round(float(predict_close), 2),
            lower_price=round(float(lower_price), 2),
            upper_price=round(float(upper_price), 2),
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
    predict_date,
    stock_code,
    actual_close,
    actual_change,
    is_correct,
):
    """
    更新指定預測的實際結果。

    以 predict_date + stock_code 尋找紀錄，
    並兼容股票代號有無 .TW / .TWO。
    """

    normalized_date = normalize_date(
        predict_date
    )

    normalized_stock_code = str(
        stock_code
    ).strip()

    db = SessionLocal()

    try:
        # 先用原始股票代號尋找
        prediction = (
            db.query(Prediction)
            .filter(
                Prediction.predict_date
                == normalized_date,
                Prediction.stock_code
                == normalized_stock_code,
            )
            .first()
        )

        # 若找不到，嘗試不同股票代號格式
        if prediction is None:
            stock_code_without_suffix = (
                normalized_stock_code
                .replace(".TW", "")
                .replace(".TWO", "")
            )

            possible_codes = [
                stock_code_without_suffix,
                stock_code_without_suffix + ".TW",
                stock_code_without_suffix + ".TWO",
            ]

            prediction = (
                db.query(Prediction)
                .filter(
                    Prediction.predict_date
                    == normalized_date,
                    Prediction.stock_code.in_(
                        possible_codes
                    ),
                )
                .first()
            )

        if prediction is None:
            print(
                "❌ 找不到待更新的 SQLite 紀錄："
                f"日期={normalized_date}，"
                f"股票={normalized_stock_code}"
            )
            return False

        prediction.actual_close = round(
            float(actual_close),
            2,
        )

        prediction.actual_change = str(
            actual_change
        )

        prediction.is_correct = str(
            is_correct
        )

        db.commit()
        db.refresh(prediction)

        print(
            "💾 SQLite 驗證結果已更新："
            f"{normalized_date} "
            f"{prediction.stock_code} "
            f"{is_correct}"
        )

        return True

    except Exception as e:
        db.rollback()

        print(
            "❌ SQLite 驗證更新異常："
            f"日期={normalized_date}，"
            f"股票={normalized_stock_code}，"
            f"原因={e}"
        )

        return False

    finally:
        db.close()   


def update_prediction_date(
    old_predict_date,
    new_predict_date,
    stock_code,
):
    """
    將指定股票的預測日期順延。

    用於颱風假、臨時休市或無交易資料時，
    更新 SQLite 中尚未驗證的預測日期。
    """

    normalized_old_date = normalize_date(
        old_predict_date
    )

    normalized_new_date = normalize_date(
        new_predict_date
    )

    normalized_stock_code = str(
        stock_code
    ).strip()

    stock_code_without_suffix = (
        normalized_stock_code
        .replace(".TW", "")
        .replace(".TWO", "")
    )

    possible_codes = [
        stock_code_without_suffix,
        stock_code_without_suffix + ".TW",
        stock_code_without_suffix + ".TWO",
    ]

    db = SessionLocal()

    try:
        prediction = (
            db.query(Prediction)
            .filter(
                Prediction.predict_date
                == normalized_old_date,
                Prediction.stock_code.in_(
                    possible_codes
                ),
                Prediction.is_correct.is_(None),
            )
            .first()
        )

        if prediction is None:
            print(
                "❌ 找不到要順延的 SQLite 紀錄："
                f"日期={normalized_old_date}，"
                f"股票={normalized_stock_code}"
            )
            return False

        prediction.predict_date = (
            normalized_new_date
        )

        db.commit()
        db.refresh(prediction)

        print(
            "💾 SQLite 預測日期已更新："
            f"{prediction.stock_code} "
            f"{normalized_old_date} "
            f"→ {normalized_new_date}"
        )

        return True

    except Exception as e:
        db.rollback()

        print(
            "❌ SQLite 預測日期更新失敗："
            f"股票={normalized_stock_code}，"
            f"日期={normalized_old_date}，"
            f"原因={e}"
        )

        return False

    finally:
        db.close()


def get_prediction_history_from_db() -> dict:
    """
    從 SQLite 取得完整預測歷史與整體勝率。
    """

    db = SessionLocal()

    try:
        predictions = (
            db.query(Prediction)
            .order_by(
                Prediction.predict_date.asc(),
                Prediction.id.asc(),
            )
            .all()
        )

        if not predictions:
            return {
                "history": [],
                "accuracy": 0,
                "validated_count": 0,
                "total_count": 0,
            }

        history = []

        correct_count = 0
        validated_count = 0

        for prediction in predictions:

            if prediction.is_correct in (
                "正確",
                "錯誤",
            ):
                validated_count += 1

                if prediction.is_correct == "正確":
                    correct_count += 1

            history.append({
                "預測日期": prediction.predict_date,
                "股票代號": prediction.stock_code,
                "股票名稱": prediction.stock_name,
                "預測結果": prediction.prediction_text,
                "信心值": prediction.confidence,
                "上漲機率": prediction.up_probability,
                "下跌機率": prediction.down_probability,
                "隔日預測參考價": prediction.predict_close,
                "預測區間下緣": prediction.lower_price,
                "預測區間上緣": prediction.upper_price,
                "實際收盤價": prediction.actual_close,
                "實際漲跌": prediction.actual_change,
                "是否預測正確": prediction.is_correct,
            })

        total_count = len(predictions)

        if validated_count > 0:
            accuracy = round(
                correct_count
                / validated_count
                * 100,
                2
            )
        else:
            accuracy = 0

        return {
            "history": history,
            "accuracy": accuracy,
            "validated_count": validated_count,
            "total_count": total_count,
        }

    except Exception as e:
        print(
            f"❌ SQLite 歷史紀錄讀取失敗：{e}"
        )

        return {
            "history": [],
            "accuracy": 0,
            "validated_count": 0,
            "total_count": 0,
        }

    finally:
        db.close() 


def get_stock_accuracy_stats_from_db() -> list:
    """
    從 SQLite 計算各股票的模型準確率。

    只統計：
    is_correct = 正確 / 錯誤
    """

    db = SessionLocal()

    try:
        predictions = (
            db.query(Prediction)
            .filter(
                Prediction.is_correct.in_(
                    ["正確", "錯誤"]
                )
            )
            .all()
        )

        if not predictions:
            return []

        stock_stats = {}

        for prediction in predictions:

            stock_code = prediction.stock_code

            if stock_code not in stock_stats:
                stock_stats[stock_code] = {
                    "stock_code": stock_code,
                    "stock_name": prediction.stock_name,
                    "correct": 0,
                    "total": 0,
                }

            stock_stats[
                stock_code
            ]["total"] += 1

            if prediction.is_correct == "正確":
                stock_stats[
                    stock_code
                ]["correct"] += 1

        result = []

        for stats in stock_stats.values():

            accuracy = round(
                stats["correct"]
                / stats["total"]
                * 100,
                2
            )

            result.append({
                "stock_code": stats["stock_code"],
                "stock_name": stats["stock_name"],
                "accuracy": accuracy,
                "correct": stats["correct"],
                "total": stats["total"],
            })

        # Accuracy 高的排前面
        # Accuracy 相同時，驗證筆數多的排前面
        result.sort(
            key=lambda item: (
                item["accuracy"],
                item["total"],
            ),
            reverse=True,
        )

        return result

    except Exception as e:
        print(
            f"❌ SQLite 股票準確率統計失敗：{e}"
        )

        return []

    finally:
        db.close()       


def get_validated_predictions_from_db() -> list:
    """
    取得所有已完成驗證的預測資料。

    只回傳：
    is_correct = 正確 / 錯誤
    """

    db = SessionLocal()

    try:
        predictions = (
            db.query(Prediction)
            .filter(
                Prediction.is_correct.in_(
                    ["正確", "錯誤"]
                )
            )
            .order_by(
                Prediction.predict_date.asc(),
                Prediction.id.asc(),
            )
            .all()
        )

        result = []

        for prediction in predictions:
            result.append({
                "predict_date": prediction.predict_date,
                "stock_code": prediction.stock_code,
                "stock_name": prediction.stock_name,
                "confidence": prediction.confidence,
                "is_correct": prediction.is_correct,
            })

        return result

    except Exception as e:
        print(
            f"❌ SQLite 已驗證資料讀取失敗：{e}"
        )

        return []

    finally:
        db.close()                  



def get_latest_prediction_from_db() -> dict | None:
    """
    取得最新一筆預測資料。
    """

    db = SessionLocal()

    try:
        prediction = (
            db.query(Prediction)
            .order_by(
                Prediction.predict_date.desc(),
                Prediction.id.desc(),
            )
            .first()
        )

        if prediction is None:
            return None

        return {
            "predict_date": prediction.predict_date,
            "stock_id": prediction.stock_code,
            "stock_name": prediction.stock_name,
            "prediction": prediction.prediction_text,
            "confidence": prediction.confidence,
            "lower_bound": prediction.lower_price,
            "upper_bound": prediction.upper_price,
            "model": "XGBoost",
        }

    except Exception as e:
        print(
            f"❌ SQLite 最新預測讀取失敗：{e}"
        )
        return None

    finally:
        db.close()


def prediction_exists_for_date_from_db(
    predict_date: str
) -> bool:
    """
    檢查指定日期是否已有任何預測紀錄。
    """

    predict_date = normalize_date(
        predict_date
    )

    db = SessionLocal()

    try:
        return (
            db.query(Prediction)
            .filter(
                Prediction.predict_date == predict_date
            )
            .first()
            is not None
        )

    except Exception as e:
        print(
            f"❌ SQLite 預測日期檢查失敗：{e}"
        )
        return False

    finally:
        db.close()


def get_accuracy_chart_data_from_db() -> dict:
    """
    從 SQLite 計算最近 30 個交易日的每日準確率。
    """

    db = SessionLocal()

    try:
        predictions = (
            db.query(Prediction)
            .filter(
                Prediction.is_correct.in_(
                    ["正確", "錯誤"]
                )
            )
            .order_by(
                Prediction.predict_date.asc(),
                Prediction.id.asc(),
            )
            .all()
        )

        if not predictions:
            return {
                "labels": [],
                "values": [],
            }

        daily_stats = {}

        for prediction in predictions:
            predict_date = normalize_date(
                prediction.predict_date
            )

            if predict_date not in daily_stats:
                daily_stats[predict_date] = {
                    "total": 0,
                    "correct": 0,
                }

            daily_stats[
                predict_date
            ]["total"] += 1

            if prediction.is_correct == "正確":
                daily_stats[
                    predict_date
                ]["correct"] += 1

        daily_result = []

        for predict_date, stats in daily_stats.items():
            accuracy = round(
                stats["correct"]
                / stats["total"]
                * 100,
                2
            )

            daily_result.append({
                "date": predict_date,
                "accuracy": accuracy,
            })

        daily_result = daily_result[-30:]

        return {
            "labels": [
                item["date"]
                for item in daily_result
            ],
            "values": [
                item["accuracy"]
                for item in daily_result
            ],
        }

    except Exception as e:
        print(
            f"❌ SQLite 勝率圖表讀取失敗：{e}"
        )

        return {
            "labels": [],
            "values": [],
        }

    finally:
        db.close()


def get_unvalidated_predictions_from_db() -> list:
    """
    取得尚未完成驗證的預測資料。
    """

    db = SessionLocal()

    try:
        predictions = (
            db.query(Prediction)
            .filter(
                Prediction.is_correct.is_(None)
            )
            .order_by(
                Prediction.predict_date.asc(),
                Prediction.id.asc(),
            )
            .all()
        )

        return [
            {
                "id": prediction.id,
                "predict_date": prediction.predict_date,
                "stock_code": prediction.stock_code,
                "stock_name": prediction.stock_name,
                "prediction_text": prediction.prediction_text,
                "predict_close": prediction.predict_close,
            }
            for prediction in predictions
        ]

    except Exception as e:
        print(
            f"❌ SQLite 未驗證預測讀取失敗：{e}"
        )
        return []

    finally:
        db.close()        
