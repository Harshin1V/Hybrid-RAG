import streamlit as st
import os
import json
import numpy as np
from datetime import datetime
import torch
from typing import List, Dict, Any

# Load dependencies for document processing
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedRAGSystem:
    """
    An advanced Retrieval-Augmented Generation (RAG) system with intelligent 
    document processing, retrieval, and feedback mechanisms.
    """
    
    def __init__(self):
        """
        Initialize the RAG system with necessary configurations and setup.
        """
        # Create necessary directories
        self._create_directories()
        
        # Initialize NLTK resources
        self._initialize_nltk()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _create_directories(self):
        """Create necessary directories for storing PDFs, models, and feedback."""
        os.makedirs("pdfs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("feedback", exist_ok=True)


    def _initialize_nltk(self):
        """Download necessary NLTK resources."""
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context  # Fix for SSL issues
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            st.success("NLTK 'punkt' downloaded successfully.")
        except Exception as e:
            st.error(f"NLTK initialization failed: {e}")
    

    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        initial_states = {
            'documents': None,
            'chunked_documents': None,
            'semantic_retriever': None,
            'bm25_retriever': None,
            'ensemble_retriever': None,
            'chat_history': [],
            'retriever_weights': [0.5, 0.5]
        }
        
        # Load existing feedback
        feedback_path = "feedback/feedback_data.json"
        if os.path.exists(feedback_path):
            with open(feedback_path, "r") as f:
                initial_states['feedback_store'] = json.load(f)
        else:
            initial_states['feedback_store'] = {}
        
        for key, value in initial_states.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @st.cache_resource
    def load_model(_self, model_type='llm', model_name=None):
        """
        Load and cache language models or embedding models.

        Args:
            model_type (str): Type of model to load ('llm' or 'embedding')
            model_name (str): Name of the model to load

        Returns:
            Loaded model instance
        """
        try:
            if model_type == 'llm':
                return OllamaLLM(model=model_name or "mistral:latest",base_url="https://916d-2406-7400-51-fd96-95e5-b4cf-9997-7336.ngrok-free.app")
            elif model_type == 'embedding':
                return HuggingFaceEmbeddings(
                    model_name=model_name or "sentence-transformers/all-mpnet-base-v2"
                )
        except Exception as e:
            st.error(f"Model loading error: {e}")
            return None
    def adjust_weights_based_on_feedback(self):
        positive_feedback = sum(1 for fb in st.session_state.feedback_store if fb['feedback'] == 'Yes')
        total_feedback = len(st.session_state.feedback_store)
        if total_feedback == 0:
            return
        positive_ratio = positive_feedback / total_feedback
        st.session_state.retriever_weights = [positive_ratio, 1 - positive_ratio]
    
    def process_pdf(self, uploaded_file, chunk_size=1000, chunk_overlap=200):
        """
        Process uploaded PDF file and prepare for retrieval.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between text chunks
        
        Returns:
            Processed document chunks
        """
        try:
            # Save PDF
            file_path = os.path.join("pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load PDF
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                add_start_index=True
            )
            chunked_documents = text_splitter.split_documents(documents)
            
            return chunked_documents
        
        except Exception as e:
            st.error(f"PDF processing error: {e}")
            return None
    
    def build_retrievers(self, documents, embedding_model, k=4):
        """
        Build semantic, BM25, and ensemble retrievers.
        
        Args:
            documents: List of document chunks
            embedding_model: Embedding model
            k (int): Number of documents to retrieve
        
        Returns:
            Tuple of (semantic_retriever, bm25_retriever, ensemble_retriever)
        """
        try:
            # Semantic retriever
            texts = [doc.page_content for doc in documents]
            metadata = [doc.metadata for doc in documents]
            vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadata)
            semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            
            # BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(documents, preprocess_func=word_tokenize, k=k)
            
            # Ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
            
            return semantic_retriever, bm25_retriever, ensemble_retriever
        
        except Exception as e:
            st.error(f"Retriever building error: {e}")
            return None, None, None
    
    def generate_answer(self, question, documents, model):
        """
        Generate answer using retrieved documents and LLM.
        
        Args:
            question (str): User's question
            documents (list): Retrieved documents
            model: Language model
        
        Returns:
            Generated answer
        """
        if not documents:
            return "No relevant documents found to answer the question."
        
        context = "\n\n".join([
            f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}): {doc.page_content}" 
            for i, doc in enumerate(documents)
        ])
        
        template = """
        You are an intelligent assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        
        try:
            return chain.invoke({"question": question, "context": context})
        except Exception as e:
            st.error(f"Answer generation error: {e}")
            return f"Error generating answer: {str(e)}"
    
    def run_rag_pipeline(self, question, embedding_model, llm_model, k_documents=4):
        """
        Run complete RAG pipeline for a given question.
        
        Args:
            question (str): User's question
            embedding_model: Embedding model
            llm_model: Language model
            k_documents (int): Number of documents to retrieve
        
        Returns:
            Generated answer and retrieved documents
        """
        self.adjust_weights_based_on_feedback()
        # Retrieve documents
        retrieved_documents = self._select_retriever(question).invoke(question)
        
        # Rerank documents
        reranked_documents = self._rerank_documents(
            question, retrieved_documents, embedding_model, top_k=k_documents
        )
        
        # Generate answer
        answer = self.generate_answer(question, reranked_documents, llm_model)
        
        return answer, reranked_documents
    
    def _select_retriever(self, question):
        """
        Dynamically select best retriever based on question type.
        
        Args:
            question (str): User's question
        
        Returns:
            Selected retriever
        """
        question_lower = question.lower()
        
        if any(term in question_lower for term in ["define", "explain", "what is", "describe"]):
            return st.session_state.semantic_retriever
        elif any(term in question_lower for term in ["when", "where", "who", "specific", "exact"]):
            return st.session_state.bm25_retriever
        
        return st.session_state.ensemble_retriever
    
    def _rerank_documents(self, question, documents, embedding_model, top_k=4):
        """
        Rerank documents using cross-attention relevance.
        
        Args:
            question (str): User's question
            documents (list): Retrieved documents
            embedding_model: Embedding model
            top_k (int): Number of top documents to return
        
        Returns:
            Reranked documents
        """
        if not documents:
            return []
        
        try:
            question_embedding = embedding_model.embed_query(question)
            doc_embeddings = [embedding_model.embed_query(doc.page_content) for doc in documents]
            
            similarities = cosine_similarity([question_embedding], doc_embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return [documents[i] for i in top_indices]
        except Exception as e:
            st.error(f"Document reranking error: {e}")
            return documents[:top_k]
    
    def collect_feedback(self, question, answer, feedback):
        feedback_path = "feedback/feedback_data.json"
        feedback_data = {
            'question': question,
            'answer': answer,
            'feedback': feedback,
            'timestamp': str(datetime.now())
        }
        if os.path.exists(feedback_path):
            with open(feedback_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(feedback_data)
        with open(feedback_path, "w") as f:
            json.dump(data, f, indent=4)
    
    def display_feedback_dashboard(self):
        feedback_path = "feedback/feedback_data.json"
        if os.path.exists(feedback_path):
            with open(feedback_path, "r") as f:
                feedback_data = json.load(f)
                st.write("### Feedback Dashboard")
                st.json(feedback_data)
        else:
            st.write("No feedback available.")



def main():
    """Main Streamlit application for Advanced RAG System"""
    st.set_page_config(page_title="Advanced RAG System", layout="wide")
    
    # Initialize RAG system
    rag_system = AdvancedRAGSystem()
    
    st.title("Advanced Document Question Answering System")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("System Configuration")
        model_name = st.selectbox("LLM Model", ["mistral:latest", "llama2:latest", "phi:latest"])
        chunk_size = st.slider("Document Chunk Size", 100, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
    
    # Document upload section
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            # Process PDF
            chunked_documents = rag_system.process_pdf(uploaded_file, chunk_size, chunk_overlap)
            
            # Load models
            embedding_model = rag_system.load_model(model_type='embedding')
            llm_model = rag_system.load_model(model_type='llm', model_name=model_name)
            
            # Build retrievers
            semantic_ret, bm25_ret, ensemble_ret = rag_system.build_retrievers(
                chunked_documents, embedding_model
            )
            
            # Update session state
            st.session_state.documents = chunked_documents
            st.session_state.semantic_retriever = semantic_ret
            st.session_state.bm25_retriever = bm25_ret
            st.session_state.ensemble_retriever = ensemble_ret
        
        st.success(f"Processed {len(chunked_documents)} document chunks")
    
    # Question answering section
    question = st.text_input("Ask a question about your document")
    
    if question and st.session_state.documents:
        with st.spinner("Generating answer..."):
            # Generate answer
            answer, used_documents = rag_system.run_rag_pipeline(
                question, 
                rag_system.load_model(model_type='embedding'),
                rag_system.load_model(model_type='llm', model_name=model_name)
            )
            
            # Display answer
            st.markdown("### Answer")
            st.write(answer)
            
            # Show used documents
            with st.expander("Retrieved Documents"):
                for i, doc in enumerate(used_documents, 1):
                    st.markdown(f"**Document {i}**")
                    st.text(doc.page_content[:500] + "...")
            feedback = st.radio("Was this answer helpful?", ["Yes", "No"])
            rag_system.collect_feedback(question, answer, feedback)
            rag_system.display_feedback_dashboard()

if __name__ == "__main__":
    main()
