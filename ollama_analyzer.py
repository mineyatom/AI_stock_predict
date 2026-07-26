"""
llm_analyzer.py

LLM 分析接口

負責：
- 建立 Prompt
- 呼叫 LLM Client

目前使用：
OpenAI

不負責：
- 股票預測
- 模型判斷
"""


from prompt_builder import (
    build_prediction_prompt
)

from llm_client import (
    generate_text
)



def analyze_prediction_with_ollama(
    prediction_data: dict
) -> str:
    """
    使用 LLM 產生股票預測解讀。

    注意：
    函式名稱保留，
    避免影響舊模組。
    """

    prompt = build_prediction_prompt(
        prediction_data
    )


    return generate_text(
        prompt
    )



def call_ollama(
    prompt: str
) -> str:
    """
    舊介面相容。

    實際呼叫由 llm_client 管理。
    """

    return generate_text(
        prompt
    )