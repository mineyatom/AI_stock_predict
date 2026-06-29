from market_summary_builder import build_market_summary_prompt
from ollama_analyzer import call_ollama


def generate_market_summary(
    market_data: dict,
    prediction_summary: dict
) -> str:
    
   
    """
    產生 AI 每日市場摘要

    Parameters
    ----------
    market_data : dict
        市場資訊

    prediction_summary : dict
        今日模型預測統計

    Returns
    -------
    str
        AI 每日市場摘要
    """

    prompt = build_market_summary_prompt(
        market_data,
        prediction_summary
    )

    summary = call_ollama(prompt)

    return summary