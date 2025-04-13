from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os

# 使用 Streamlit Secrets 来获取 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def ask_doctor_ai(query):
    # 加载当前上传的病历文本
    loader = TextLoader("data/current.txt")
    docs = loader.load()
    
    # 将文档拆分为可处理的片段
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    
    # 从拆分后的文档构建向量数据库，注意 persist_directory 用于数据存储
    vectordb = Chroma.from_documents(splits, OpenAIEmbeddings(), persist_directory="vectorstore")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # 使用 ChatGPT 模型进行问答，chain_type 设置为 "stuff" 简单拼装答案
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=retriever
    )
    
    result = qa.run(query)
    return result
