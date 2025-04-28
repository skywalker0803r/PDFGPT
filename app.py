import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile

# 1. Streamlit 介面
st.title("📚 GPT PDF 問答機 (輸入金鑰版)")
st.write("上傳 PDF，輸入 OpenAI 金鑰，開始問問題！")

# 2. 要求使用者輸入 OpenAI API Key
openai_api_key = st.text_input("請輸入你的 OpenAI API 金鑰：", type="password")

# 3. 上傳 PDF
uploaded_file = st.file_uploader("選擇一份 PDF 檔案", type="pdf")

if openai_api_key and uploaded_file is not None:
    # 設定環境變數 (臨時)
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # 把上傳的檔案存到臨時檔案
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # 讀取 PDF
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # 切割成小段落
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # 建立向量資料庫
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    # 初始化 LLM
    llm = ChatOpenAI(model="gpt-4o")  # 或 gpt-3.5-turbo
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )

    # 問問題
    query = st.text_input("請輸入你的問題：")

    if st.button("送出問題"):
        if query:
            with st.spinner("思考中..."):
                result = qa.invoke(query)
                st.success(result["result"])
        else:
            st.warning("請先輸入一個問題！")
else:
    st.info("請先輸入 OpenAI API 金鑰，並上傳 PDF 檔案。")
