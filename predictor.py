from unittest import result


def predict_stock(stock_id):

    result = {
        "stock_id": stock_id,
        "direction": "上漲",
        "confidence": 63.5,
        "up_probability": 63.5,
        "down_probability": 36.5,
        "latest_close": 875,
        "price_range": "875 ~ 902"
    }

    return result 