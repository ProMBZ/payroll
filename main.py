import streamlit as st
import re
import os
from io import StringIO
from dotenv import load_dotenv
from datetime import datetime
import speech_recognition as sr
import PyPDF2
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Load .env variables
load_dotenv()

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GEMINI_API_KEY
)

# Memory initialization
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Streamlit UI setup
st.set_page_config(page_title="💸 Paycheck Analyzer with Voice & Chat")
st.title("💸 Paycheck Analyzer with Voice & Chat")

# Step 1: Choose input method
st.subheader("Step 1: Upload your Pay Stub or Enter Text")
input_method = st.radio("Choose input method:", ("Upload File", "Manual Text", "Voice Input"))

paycheck_text = ""

if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload Pay Stub (Text or PDF)", type=["txt", "pdf"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            # Read PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            paycheck_text = ""
            for page in pdf_reader.pages:
                paycheck_text += page.extract_text()
        else:
            # Read Text file
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            paycheck_text = stringio.read()

elif input_method == "Manual Text":
    paycheck_text = st.text_area("Paste paycheck text here")

elif input_method == "Voice Input":
    st.info("Click to record your voice. Please talk clearly.")
    if st.button("🎙 Start Recording"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = recognizer.listen(source)
            try:
                paycheck_text = recognizer.recognize_google(audio)
                st.success("Voice input recognized:")
                st.write(paycheck_text)
            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")

# Analyze Button
if st.button("🔍 Analyze Paycheck") and paycheck_text:
    st.subheader("🔎 Analysis Result")

    prompt = f"""
    Analyze the following paycheck text. Extract hours worked, hourly rate, total gross pay, total deductions, and net pay.
    Also, explain each value simply like you are talking to someone new to paychecks:

    {paycheck_text}
    """

    try:
        response = llm.invoke(prompt)
        explanation = response.content
        st.markdown(explanation)
        memory.save_context({"input": prompt}, {"output": explanation})
    except Exception as e:
        st.error(f"Error analyzing paycheck: {e}")

# Chatbot interface using Gemini
st.subheader("💬 Ask a Question About Your Paycheck")
user_question = st.text_input("Ask a question")
if st.button("Ask") and user_question:
    chat_prompt = f"The user has this paycheck info: {paycheck_text}\n\nQuestion: {user_question}"
    try:
        response = llm.invoke(chat_prompt)
        answer = response.content
        st.success(answer)
        memory.save_context({"input": chat_prompt}, {"output": answer})
    except Exception as e:
        st.error(f"Error: {e}")
