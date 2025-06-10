import streamlit as st
import json
import os
import hashlib
import time
import nltk
from student_bot import predict_class, get_response
from teacher_bot import ask_teacher_model

nltk.download('punkt')
nltk.download('wordnet')

with open('data/intents.json') as f:
    intents = json.load(f)


def store_new_intent(pattern, response):
    new_intents_path = "data/intents.json"

    if not os.path.exists(new_intents_path):
        with open(new_intents_path, "w") as f:
            json.dump({"intents": []}, f, indent=4)

    with open(new_intents_path, "r") as f:
        data = json.load(f)

    tag_hash = hashlib.md5(response.encode()).hexdigest()[:8]
    tag = f"auto_{tag_hash}"

    for intent in data["intents"]:
        if response.strip() in intent["responses"]:
            if pattern not in intent["patterns"]:
                intent["patterns"].append(pattern)
            break
    else:
        data["intents"].append({
            "tag": tag,
            "patterns": [pattern],
            "responses": [response]
        })

    with open(new_intents_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def chat(user_input):
    predictions = predict_class(user_input)

    if predictions and float(predictions[0]['probability']) > 0.95:
        response = get_response(predictions, intents)
        time.sleep(5)
    else:
        response = ask_teacher_model(user_input)
        store_new_intent(user_input, response)

    return response


st.set_page_config(page_title="Alissa Chatbot", page_icon="🤖", layout="centered")

st.title("Alissa Chatbot")
st.caption("Chat with the AI Assistant powered by student-teacher models")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type a message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response = chat(user_input)
    st.session_state.messages.append({"role": "assistant", "content": f"🤖 Alissa: {response}"})
    with st.chat_message("assistant"):
        st.markdown(f"Alissa: {response}")
