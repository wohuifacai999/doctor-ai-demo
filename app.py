import streamlit as st
from langchain_chain import ask_doctor_ai
import os

st.set_page_config(page_title="DoctorMind.AI", layout="wide")
st.title("🧠 DoctorMind.AI - 医疗智能助理")

uploaded_file = st.file_uploader("上传病历（TXT 或 PDF）", type=["txt", "pdf"])
user_question = st.text_input("你有什么疑问想问AI？")

if uploaded_file:
    content = uploaded_file.read().decode('utf-8', errors='ignore')
    with open("data/current.txt", "w") as f:
        f.write(content)
    st.success("病历上传成功 ✅")

if st.button("生成回答") and user_question:
    with st.spinner("AI正在思考中..."):
        response = ask_doctor_ai(user_question)
        st.markdown("#### 🧠 AI回答：")
        st.markdown(response)
