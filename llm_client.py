"""
llm_client.py

統一管理外部 LLM

目前使用：
OpenAI GPT-5-mini

用途：
只負責文字生成
不負責股票預測
"""


import os

from openai import OpenAI



client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY"
    )
)


MODEL_NAME = "gpt-5-mini"



def generate_text(
    prompt: str
) -> str:
    """
    呼叫 OpenAI 產生文字。

    失敗時回傳 fallback，
    不影響股票預測流程。
    """

    try:

        response = client.responses.create(

            model=MODEL_NAME,

            input=prompt,

            timeout=30,

        )


        return (
            response
            .output_text
            .strip()
        )


    except Exception as e:

        print(
            "[WARN] OpenAI API 錯誤：",
            e
        )


        return (
            "AI模型摘要暫時無法產生。\n"
            "模型預測結果仍以機器學習模型輸出為準。"
        )