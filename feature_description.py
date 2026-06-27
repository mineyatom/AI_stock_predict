FEATURE_DESCRIPTION = {
    "RSI": {
        "name": "RSI 動能指標",
        "meaning": "用來觀察股價近期動能是否偏強或偏弱。"
    },
    "KD-K值": {
        "name": "KD-K 短線動能",
        "meaning": "反映短線買賣動能變化。"
    },
    "KD-D值": {
        "name": "KD-D 短線趨勢",
        "meaning": "用來輔助判斷短線趨勢是否延續。"
    },
    "當日報酬率": {
        "name": "當日股價表現",
        "meaning": "反映股價當日漲跌對模型的影響。"
    },
    "前一日報酬率": {
        "name": "前一日股價變化",
        "meaning": "用來觀察前一交易日價格變化是否影響隔日方向。"
    },
    "前二日報酬率": {
        "name": "前二日股價變化",
        "meaning": "反映近期價格慣性或反轉訊號。"
    },
    "前三日報酬率": {
        "name": "近三日股價動能",
        "meaning": "用來觀察短期股價是否有連續動能。"
    },
    "成交量均線": {
        "name": "成交量趨勢",
        "meaning": "反映市場交易熱度是否增加或減弱。"
    },
    "外資買賣超": {
        "name": "外資籌碼變化",
        "meaning": "反映外資近期買賣方向對模型的影響。"
    },
    "投信買賣超": {
        "name": "投信籌碼變化",
        "meaning": "反映投信近期布局方向。"
    },
    "自營商買賣超": {
        "name": "自營商籌碼變化",
        "meaning": "反映自營商短線操作方向。"
    },
    "避險自營商": {
        "name": "避險自營商部位變化",
        "meaning": "反映避險部位對短線市場判斷的影響。"
    },
    "大盤報酬率": {
        "name": "台股大盤走勢",
        "meaning": "反映整體市場方向對個股的影響。"
    },
    "大盤RSI": {
        "name": "大盤動能狀態",
        "meaning": "用來觀察整體市場動能是否偏強或偏弱。"
    },
    "大盤波動率": {
        "name": "大盤波動風險",
        "meaning": "反映整體市場震盪程度。"
    },
    "NVIDIA漲跌幅": {
        "name": "NVIDIA 股價變化",
        "meaning": "反映 AI 與半導體族群的外部市場情緒。"
    },
    "SOX漲跌幅": {
        "name": "費半指數漲跌幅",
        "meaning": "反映全球半導體族群走勢。"
    },
    "QQQ漲跌幅": {
        "name": "NASDAQ 科技股 ETF 變化",
        "meaning": "反映美國科技股整體市場情緒。"
    },
}


def explain_feature(feature_name):
    data = FEATURE_DESCRIPTION.get(feature_name)

    if data is None:
        return {
            "name": feature_name,
            "meaning": "此特徵為模型使用的輸入變數。"
        }

    return data


def explain_feature_list(features):
    if not features:
        return []

    result = []

    for feature in features:
        data = explain_feature(feature)

        result.append(
            f"{data['name']}：{data['meaning']}"
        )

    return result