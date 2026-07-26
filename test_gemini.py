import os
from google import genai


api_key = os.getenv("GEMINI_API_KEY")


print("API KEY 是否存在：", bool(api_key))


genai.configure(
    api_key=api_key
)


model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


response = model.generate_content(
    "請用一句話介紹你自己"
)


print(response.text)