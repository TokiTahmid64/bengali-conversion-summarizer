import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

PROMPT = """You are a Bengali conversation summarizer application. Your job is to extract key points from a conversation text. You will extract any promises, agreements and notable conversation matters. Below is the conversation text in Bengali. Your output will be in Bengali. Extract the key points in a list format:
----------------------------------------------------------------------------------------------------"""


def summarize_conversation(conversation: str) -> str:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-pro")
    inp = f"{PROMPT}\n{conversation}"
    response = model.generate_content(inp)
    return response.text
