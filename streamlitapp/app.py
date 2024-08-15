import streamlit as st
import threading
from rag_workflow import RAGWorkflow
from chat_interface import ChatInterface
from pdf_processor import PDFProcessor
from model_manager import ModelManager
from pathlib import Path
import os
import logging


# Define base directory for the app
BASE_DIR = Path.home() / ".rag_app"

# Create directories if they don't exist
BASE_DIR.mkdir(exist_ok=True)
(BASE_DIR / "pdfs").mkdir(exist_ok=True)
(BASE_DIR / "documents_db").mkdir(exist_ok=True)
(BASE_DIR / "questions_db").mkdir(exist_ok=True)

# Define paths
PDF_DIRECTORY = str(BASE_DIR / "pdfs")
CHROMA_DOCUMENTS_DIRECTORY = str(BASE_DIR / "documents_db")
CHROMA_QUESTIONS_DIRECTORY = str(BASE_DIR / "questions_db")

st.set_page_config(layout="wide")

# Initialize session state
if 'rag_workflow' not in st.session_state:
    # st.session_state.rag_workflow = RAGWorkflow()
    st.session_state.rag_workflow = RAGWorkflow(PDF_DIRECTORY, CHROMA_DOCUMENTS_DIRECTORY, CHROMA_QUESTIONS_DIRECTORY)
if 'chat_interface' not in st.session_state:
    st.session_state.chat_interface = ChatInterface()
if 'pdf_processor' not in st.session_state:
    # st.session_state.pdf_processor = PDFProcessor()
    st.session_state.pdf_processor = PDFProcessor(PDF_DIRECTORY, CHROMA_DOCUMENTS_DIRECTORY, st.session_state.rag_workflow.vectorstore)
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Layout
left_column, right_column = st.columns([1, 2])

with left_column:
    # PDF Upload
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type='pdf')
    
    # if uploaded_files:
    #     with st.spinner('Processing PDFs...'):
    #         thread = threading.Thread(target=st.session_state.pdf_processor.process_pdfs, args=(uploaded_files,))
    #         thread.start()

    if uploaded_files:
        with st.spinner('Processing PDFs...'):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(PDF_DIRECTORY, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                logging.info(f"Saved file: {file_path}")

            if st.session_state.pdf_processor.process_pdfs():
                st.session_state.processing_complete = True
                st.success("PDFs processed successfully!")
                st.session_state.rag_workflow.generate_questions()
                st.experimental_rerun()

            # thread = threading.Thread(target=st.session_state.pdf_processor.process_pdfs, args=(PDF_DIRECTORY,))
            # thread.start()
    
    # Generated Questions
    st.subheader("Generated Questions")
    questions = st.session_state.rag_workflow.get_generated_questions()
    logging.info(f"Retrieved {len(questions)} questions")
    for question in questions:
        if st.button(question):
            st.session_state.chat_interface.add_user_message(question)

with right_column:
    # Model Selection and New Chat
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox("Select Model", ["llama3.1", "llama3.1:70b", "llama3.1:405b"])
        st.session_state.model_manager.set_model(selected_model)
    with col2:
        if st.button("New Chat"):
            st.session_state.chat_interface.clear_chat()
    
    # Chat Interface
    st.session_state.chat_interface.display_chat()
    
    user_input = st.text_input("Type your message here...")
    if user_input:
        response = st.session_state.rag_workflow.get_response(user_input)
        st.session_state.chat_interface.add_user_message(user_input)
        st.session_state.chat_interface.add_ai_message(response)

# Run the app
# if __name__ == "__main__":
#     st.run()