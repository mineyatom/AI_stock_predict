FEATURE_DESCRIPTION = {
    "RSI": "RSI 動能指標",
    "KD-K值": "KD-K 短線動能",
    "KD-D值": "KD-D 短線趨勢",
    "MACD": "MACD 趨勢動能",
    "MACD_Signal": "MACD 訊號線",
    "MACD_Diff": "MACD 多空差距",

    "當日報酬率": "當日股價表現",
    "前一日報酬率": "前一日股價變化",
    "前二日報酬率": "前二日股價變化",
    "前三日報酬率": "近三日股價動能",
    "波動率": "近期價格波動程度",
    "成交量均線": "成交量趨勢",

    "外資買賣超": "外資籌碼變化",
    "投信買賣超": "投信籌碼變化",
    "自營商買賣超": "自營商籌碼變化",
    "避險自營商": "避險自營商部位變化",

    "大盤報酬率": "台股大盤走勢",
    "大盤RSI": "大盤動能狀態",
    "大盤波動率": "大盤波動風險",

    "NVIDIA漲跌幅": "NVIDIA 股價變化",
    "SOX漲跌幅": "費城半導體指數變化",
    "QQQ漲跌幅": "NASDAQ 科技股 ETF 變化",
}


def explain_feature(feature_name):
    return FEATURE_DESCRIPTION.get(
        feature_name,
        feature_name
    )


def explain_feature_list(features):
    if not features:
        return []

    return [
        explain_feature(feature)
        for feature in features
    ]