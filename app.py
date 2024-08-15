import streamlit as st

st.set_page_config(page_title="PDF Question Answering System", layout="wide")

import os
import tempfile
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
import json
import re
from typing import List, Dict

# Assuming you have already defined the necessary functions like load_documents, generate_questions, etc.

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = []

# Streamlit UI
st.title("PDF Question Answering System")

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Add the generate_questions function here
def extract_questions_fallback(text: str) -> List[Dict]:
    """
    Fallback method to extract questions when JSON parsing fails.
    """
    questions = []
    pattern = r'(?:Question:|^\d+\.)\s*(.+?)\s*Difficulty:\s*(\w+)'
    matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
    
    for i, (question, difficulty) in enumerate(matches, 1):
        questions.append({
            "question": question.strip(),
            "difficulty": difficulty.strip().lower(),
        })
    
    return questions

def generate_questions(documents):
    llm = ChatOllama(
        model="llama3.1",
        temperature=0.7,
    )

    question_prompt = PromptTemplate(
        template="""You are an expert at creating insightful questions based on given text.
        Read the following text and generate 3 relevant questions that can be answered using the information provided.
        Make sure the questions are diverse and cover different aspects of the text.
        
        Text: {text}
        
        Output the questions in the following format:
        Question 1: [Question text]
        Difficulty: [easy/medium/hard]

        Question 2: [Question text]
        Difficulty: [easy/medium/hard]

        Question 3: [Question text]
        Difficulty: [easy/medium/hard]
        
        Generated Questions:""",
        input_variables=["text"],
    )

    question_chain = LLMChain(llm=llm, prompt=question_prompt, output_key="questions")

    all_questions = []

    for doc in documents:
        result = question_chain.invoke({"text": doc.page_content})
        questions_text = result["questions"]
        
        try:
            # First, try to parse as JSON (in case the LLM outputs JSON despite the new format)
            questions = json.loads(questions_text)
            if isinstance(questions, dict) and "questions" in questions:
                questions = questions["questions"]
        except json.JSONDecodeError:
            # If JSON parsing fails, use the fallback method
            questions = extract_questions_fallback(questions_text)
        
        for q in questions:
            q["pdf_name"] = doc.metadata.get("source", "Unknown")
            q["page_number"] = doc.metadata.get("page", "Unknown")
        
        all_questions.extend(questions)

    return all_questions

if uploaded_files:
    # Create a temporary directory to store uploaded PDFs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to the temporary directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Load documents
        @st.cache_data
        def load_documents(directory):
            document_loader = PyPDFDirectoryLoader(directory)
            return document_loader.load()

        st.session_state.documents = load_documents(temp_dir)

        # Process documents
        if st.session_state.documents:
            st.success(f"Successfully loaded {len(st.session_state.documents)} documents.")

            # Initialize text splitter
            
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=250, chunk_overlap=0
            )
            st.success(f"Successfully Split the text.")

            # Split the documents into chunks
            doc_splits = text_splitter.split_documents(st.session_state.documents)
            st.success(f"Successfully Split the documents into chunks.")

            # Create vector store
            st.session_state.vectorstore = SKLearnVectorStore.from_documents(
                documents=doc_splits,
                embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
            )
            st.success(f"Successfully Created a vector store.")

            # Generate questions
            if st.button("Generate Questions"):
                with st.spinner("Generating questions..."):
                    st.session_state.generated_questions = generate_questions(st.session_state.documents)
                st.success("Questions generated successfully!")

            # Display generated questions
            if st.session_state.generated_questions:
                st.subheader("Generated Questions")
                for q in st.session_state.generated_questions:
                    st.write(f"Question: {q['question']}")
                    st.write(f"Difficulty: {q['difficulty']}")
                    st.write(f"PDF: {q['pdf_name']}")
                    st.write(f"Page: {q['page_number']}")
                    st.write("---")

# Chat interface
st.subheader("Ask a Question")
user_question = st.text_input("Enter your question here:")

if user_question and st.session_state.vectorstore:
    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            # Retrieve relevant documents
            retriever = st.session_state.vectorstore.as_retriever(k=4)
            relevant_docs = retriever.invoke(user_question)

            # Generate answer
            llm = ChatOllama(model="llama3.1", temperature=0)
            prompt = PromptTemplate(
                template="""You are an assistant for question-answering tasks. 
                Use the following documents to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Use three sentences maximum and keep the answer concise:
                Question: {question} 
                Documents: {documents} 
                Answer: """,
                input_variables=["question", "documents"],
            )
            rag_chain = prompt | llm | StrOutputParser()
            
            answer = rag_chain.invoke({"documents": relevant_docs, "question": user_question})

            # Display the answer
            st.subheader("Answer")
            st.write(answer)

            # Display source documents
            st.subheader("Source Documents")
            for i, doc in enumerate(relevant_docs):
                st.write(f"Document {i+1}:")
                st.write(f"Content: {doc.page_content}...")
                st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
                st.write(f"Page: {doc.metadata.get('page', 'Unknown')}")
                st.write("---")



# Run the Streamlit app
if __name__ == "__main__":
    pass