import streamlit as st
import os
from utils import load_pdf, split_docs, create_vectorstore
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 LangChain PDF Chatbot")

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 1: File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    st.info("Processing PDF...")
    docs = load_pdf("temp.pdf")
    chunks = split_docs(docs)
    vectorstore = create_vectorstore(chunks)
    
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    
    # Step 2: Chat input
    question = st.text_input("Ask something about the PDF:")
    if question:
        result = qa_chain({"question": question, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((question, result["answer"]))
    
    # Step 3: Display chat
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
          
