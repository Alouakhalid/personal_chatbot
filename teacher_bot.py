import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key="AIzaSyDU_paXZtjtzpGd0XV4RBfCwLi1cLnyhfw")


model = genai.GenerativeModel(model_name="gemini-2.0-flash")

def ask_teacher_model(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error {e}"
