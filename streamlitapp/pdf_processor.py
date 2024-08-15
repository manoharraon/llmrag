import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
import os
import tempfile
import logging

class PDFProcessor:
    def __init__(self, pdf_directory, chroma_documents_directory):
        self.pdf_directory = pdf_directory
        self.embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        self.chroma_documents_directory = chroma_documents_directory
        self.documents_collection_name = "pdf_documents"
        self.vectorstore = self.get_or_create_vectorstore()

    def get_or_create_vectorstore(self):
        return Chroma(
            persist_directory=self.chroma_documents_directory,
            embedding_function=self.embeddings,
            collection_name=self.documents_collection_name
        )

    # def process_pdfs(self, uploaded_files):
    #     for uploaded_file in uploaded_files:
    #         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    #             tmp_file.write(uploaded_file.getvalue())
    #             tmp_file_path = tmp_file.name

    #         loader = PyPDFLoader(tmp_file_path)
    #         pages = loader.load_and_split()

    #         text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #             chunk_size=250, chunk_overlap=0
    #         )
    #         splits = text_splitter.split_documents(pages)

    #         self.vectorstore.add_documents(splits)
    #         self.vectorstore.persist()

    #         os.unlink(tmp_file_path)

    #     st.success("PDFs processed and added to the vector store!")
    def process_pdfs(self, pdf_directory):
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_directory, filename)
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()

                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=250, chunk_overlap=0
                )
                splits = text_splitter.split_documents(pages)

                self.vectorstore.add_documents(splits)
        
        self.vectorstore.persist()
        logging.info("PDFs processed and added to the vector store")
        return True
        # st.success("PDFs processed and added to the vector store!")