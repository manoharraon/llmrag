import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
import os
import json
import uuid
import re
from typing import TypedDict, List
from dotenv import load_dotenv

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class GraphState(TypedDict):
    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]
    source: str

class RAGWorkflow:
    def __init__(self, pdf_directory, chroma_documents_directory, chroma_questions_directory):
        self.pdf_directory = pdf_directory
        self.chroma_documents_directory = chroma_documents_directory
        self.chroma_questions_directory = chroma_questions_directory
        self.documents_collection_name = "pdf_documents"
        self.questions_collection_name = "generated_questions"
        self.embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        self.vectorstore = self.get_or_create_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.llm = ChatOllama(model="llama3.1", temperature=0)
        self.web_search_tool = TavilySearchResults()
        self.workflow = self.create_workflow()

    def get_or_create_vectorstore(self):
        if os.path.exists(self.chroma_documents_directory):
            return Chroma(persist_directory=self.chroma_documents_directory, 
                          embedding_function=self.embeddings, 
                          collection_name=self.documents_collection_name)
        return Chroma.from_documents(
            documents=self.load_and_process_documents(),
            embedding=self.embeddings,
            persist_directory=self.chroma_documents_directory,
            collection_name=self.documents_collection_name
        )

    def load_and_process_documents(self):
        loader = PyPDFDirectoryLoader(self.pdf_directory)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        return text_splitter.split_documents(documents)

    def generate_questions(self, document):
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
        question_chain = LLMChain(llm=self.llm, prompt=question_prompt, output_key="questions")
        result = question_chain.invoke({"text": document.page_content})
        questions = self.parse_questions(result["questions"])
        for q in questions:
            q["pdf_name"] = document.metadata.get("source", "Unknown")
            q["page_number"] = document.metadata.get("page", "Unknown")
        return questions

    def parse_questions(self, questions_text):
        questions = []
        pattern = r'Question\s+(\d+):\s+(.*?)\s+Difficulty:\s+(easy|medium|hard)'
        matches = re.findall(pattern, questions_text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            question_number, question_text, difficulty = match
            questions.append({
                "question": question_text.strip(),
                "difficulty": difficulty.lower(),
                "number": int(question_number)
            })
        
        # If no matches found, attempt a fallback parsing method
        if not questions:
            lines = questions_text.split('\n')
            current_question = {}
            for line in lines:
                line = line.strip()
                if line.startswith("Question"):
                    if current_question:
                        questions.append(current_question)
                    current_question = {"question": line.split(":", 1)[1].strip()}
                elif line.startswith("Difficulty:"):
                    current_question["difficulty"] = line.split(":", 1)[1].strip().lower()
            if current_question:
                questions.append(current_question)
        
        # Assign numbers if they're missing
        for i, q in enumerate(questions, 1):
            if "number" not in q:
                q["number"] = i
        
        return questions

    def get_generated_questions(self):
        questions_db = Chroma(persist_directory=self.chroma_questions_directory, 
                              embedding_function=self.embeddings, 
                              collection_name=self.questions_collection_name)
        results = questions_db.get()
        all_questions = []
        for doc in results['documents']:
            all_questions.extend(json.loads(doc))
        return [q["question"] for q in all_questions]

    # def create_workflow(self):
    #     # Implement the workflow creation logic here
    #     # This should include the retrieve, grade_documents, generate, and web_search functions
    #     # as well as the StateGraph setup
    #     pass
    def create_workflow(self):
    # Define the retrieval function
        def retrieve(state):
            question = state["question"]
            documents = self.retriever.invoke(question)
            steps = state["steps"]
            steps.append("retrieve_documents")
            return {"documents": documents, "question": question, "steps": steps}

        # Define the document grading function
        def grade_documents(state):
            question = state["question"]
            documents = state["documents"]
            steps = state["steps"]
            steps.append("grade_document_retrieval")
            
            retrieval_grader_prompt = PromptTemplate(
                template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
                Here is the retrieved document: \n\n {document} \n\n
                Here is the user question: {question} \n
                If the document contains keywords related to the user question, grade it as relevant. \n
                It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
                input_variables=["question", "document"],
            )
            retrieval_grader = retrieval_grader_prompt | self.llm | JsonOutputParser()

            filtered_docs = []
            search = "No"
            for d in documents:
                score = retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                grade = score["score"]
                if grade == "yes":
                    filtered_docs.append(d)
                else:
                    search = "Yes"
            
            return {
                "documents": filtered_docs,
                "question": question,
                "search": search,
                "steps": steps,
            }

        # Define the generation function
        def generate(state):
            question = state["question"]
            documents = state["documents"]
            steps = state["steps"]

            source = "WebSearch" if "web_search" in steps else "RAG"
            
            rag_chain_prompt = PromptTemplate(
                template="""You are an assistant for question-answering tasks. 
                Use the following documents to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Use three sentences maximum and keep the answer concise:
                Question: {question} 
                Documents: {documents} 
                Answer: """,
                input_variables=["question", "documents"],
            )
            rag_chain = rag_chain_prompt | self.llm | StrOutputParser()
            
            generation = rag_chain.invoke({"documents": documents, "question": question})
            steps.append("generate_answer")
            return {
                "documents": documents,
                "question": question,
                "generation": generation,
                "steps": steps,
                "source": source,
            }

        # Define the web search function
        def web_search(state):
            question = state["question"]
            documents = state.get("documents", [])
            steps = state["steps"]
            steps.append("web_search")
            web_results = self.web_search_tool.invoke({"query": question})
            documents.extend(
                [
                    Document(page_content=d["content"], metadata={"url": d["url"]})
                    for d in web_results
                ]
            )
            return {"documents": documents, "question": question, "steps": steps}

        # Define the decision function
        def decide_to_generate(state):
            search = state["search"]
            if search == "Yes":
                return "search"
            else:
                return "generate"

        # Create the workflow
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("web_search", web_search)

        # Set the entrypoint
        workflow.set_entry_point("retrieve")

        # Add edges
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "search": "web_search",
                "generate": "generate",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def get_response(self, question):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state_dict = self.workflow.invoke({"question": question, "steps": []}, config)
        
        output = {
            "response": state_dict["generation"],
            "steps": state_dict["steps"],
            "source": state_dict["source"]
        }
        
        if state_dict["source"] == "RAG":
            relevant_docs = []
            for doc in state_dict["documents"]:
                doc_info = {
                    "content": doc.page_content,
                    "file_name": doc.metadata.get("source", "Unknown"),
                    "page_number": doc.metadata.get("page", "Unknown")
                }
                relevant_docs.append(doc_info)
            output["relevant_documents"] = relevant_docs
        
        return output