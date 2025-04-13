from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

os.environ["OPENAI_API_KEY"] = "你的API密钥"  # <<<<<< 这里替换成你的 API Key

def ask_doctor_ai(query):
    loader = TextLoader("data/current.txt")
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    
    vectordb = Chroma.from_documents(splits, OpenAIEmbeddings(), persist_directory="vectorstore")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
                                     chain_type="stuff",
                                     retriever=retriever)
    result = qa.run(query)
    return result
