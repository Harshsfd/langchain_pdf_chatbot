# main.py
import streamlit as st
import os
import tempfile

# Try to import load_dotenv safely (so app doesn't crash if dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv isn't installed, we continue — Streamlit Cloud installs from requirements.txt.
    pass

from utils import load_pdf, split_docs, create_vectorstore

# LangChain imports (chat model + chain)
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 LangChain PDF Chatbot")

# Initialize session state items
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer) tuples

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.source_filename = None

# Sidebar: model settings and API key fallback
st.sidebar.header("Settings")
openai_key = os.environ.get("OPENAI_API_KEY", "")
if not openai_key:
    # optional: allow user to paste API key in UI (not recommended for production)
    openai_key_input = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
    if openai_key_input:
        os.environ["OPENAI_API_KEY"] = openai_key_input

model_name = st.sidebar.selectbox("Model", options=["gpt-3.5-turbo", "gpt-4"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Re-process file if a new file is uploaded
if uploaded_file is not None:
    # Use a temp file to save uploaded contents so pypdf can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # If we already processed this same file (by name), skip re-processing:
    filename = getattr(uploaded_file, "name", None) or os.path.basename(tmp_path)
    # If different file, or vectorstore missing, process
    if st.session_state.vectorstore is None or st.session_state.source_filename != filename:
        st.info("Processing PDF — extracting text and creating vector index (this may take a few seconds)...")
        try:
            docs = load_pdf(tmp_path)
            chunks = split_docs(docs, chunk_size=1000, chunk_overlap=200)
            vectorstore = create_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore
            st.session_state.source_filename = filename
            st.success("Document processed. You can now ask questions.")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
else:
    st.info("Upload a PDF to begin. (You can paste OpenAI API key in the sidebar if not set in environment.)")

# If vectorstore exists, show chat UI
if st.session_state.vectorstore is not None:
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    # initialize LLM
    try:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        llm = None

    if llm is not None:
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=True)

        # Input box
        question = st.text_input("Ask something about the PDF:", key="question_input")
        if question:
            with st.spinner("Thinking..."):
                try:
                    result = qa_chain({"question": question, "chat_history": st.session_state.chat_history})
                    answer = result.get("answer") or result.get("result") or ""
                    # append to session history
                    st.session_state.chat_history.append((question, answer))
                    # show answer
                    st.markdown("**Answer:**")
                    st.write(answer)
                    # show sources if present
                    source_docs = result.get("source_documents")
                    if source_docs:
                        st.markdown("**Source documents / pages:**")
                        for sd in source_docs:
                            # Each source doc has .metadata — show source and page if present
                            meta = sd.metadata or {}
                            src = meta.get("source", "unknown")
                            page = meta.get("page")
                            if page:
                                st.write(f"- {src} (page {page})")
                            else:
                                st.write(f"- {src}")
                except Exception as e:
                    st.error(f"Error during QA: {e}")

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for q, a in reversed(st.session_state.chat_history):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")
                st.write("---")
