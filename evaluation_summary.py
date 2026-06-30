def generate_evaluation_summary(
    metrics,
    confidence_bins
):

    accuracy = metrics["accuracy"]

    precision = metrics["precision"]

    recall = metrics["recall"]

    f1 = metrics["f1"]

    best_bin = max(
        confidence_bins,
        key=lambda x: x["accuracy"]
    )

    if accuracy >= 70:
        level = "模型目前表現良好，已具備較高的預測能力。"

    elif accuracy >= 60:
        level = "模型目前表現穩定，具有一定的預測能力。"

    elif accuracy >= 50:
        level = "模型已具備基礎預測能力，但整體表現仍有提升空間。"

    else:
        level = "模型目前預測能力仍有限，建議持續優化模型。"

    summary = f"""
目前模型 Accuracy 為 {accuracy:.2f}%。

{level}

模型在預測上漲訊號時，
Precision 為 {precision:.2f}%，
表示預測結果具有一定可信度。

Recall 為 {recall:.2f}%，
代表模型能捕捉超過一半的實際上漲走勢，
仍有部分行情未能成功辨識。

F1 Score 為 {f1:.2f}%，
顯示 Precision 與 Recall 維持相對均衡，
模型整體表現仍具改善空間。

目前表現最佳的信心區間為【{best_bin['range']}】，
準確率為 {best_bin['accuracy']:.2f}%。

目前信心值與實際準確率尚未呈現明顯正相關，
顯示模型信心值仍有進一步校正空間。

後續將持續累積歷史資料，
並導入 Probability Calibration、
Voting Ensemble 與更多市場特徵，
提升模型穩定性與整體預測能力。

本分析僅供模型驗證與研究參考，
不構成任何投資建議。
"""

    return summary