import os

from openai import OpenAI


client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY"
    )
)


response = client.responses.create(
    model="gpt-5-mini",
    input="請用一句話介紹你自己"
)


print(response.output_text)