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

# Ensure directories exist
os.makedirs("pdfs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("feedback", exist_ok=True)

# Initialize NLTK
nltk.download('punkt', quiet=True)

# Streamlit app configuration
st.set_page_config(page_title="Advanced RAG with Feedback", layout="wide")

# Session state initialization
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'chunked_documents' not in st.session_state:
    st.session_state.chunked_documents = None
if 'semantic_retriever' not in st.session_state:
    st.session_state.semantic_retriever = None
if 'bm25_retriever' not in st.session_state:
    st.session_state.bm25_retriever = None
if 'ensemble_retriever' not in st.session_state:
    st.session_state.ensemble_retriever = None
if 'feedback_store' not in st.session_state:
    # Load existing feedback if available
    feedback_path = "feedback/feedback_data.json"
    if os.path.exists(feedback_path):
        with open(feedback_path, "r") as f:
            st.session_state.feedback_store = json.load(f)
    else:
        st.session_state.feedback_store = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Model selection
@st.cache_resource
def load_ollama_model(model_name):
    """Load Ollama model with caching to improve performance"""
    try:
        return OllamaLLM(model=model_name)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Embedding model selection
@st.cache_resource
def load_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    """Load embedding model with caching"""
    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# PDF processing functions
def upload_pdf(file):
    """Save uploaded PDF to disk"""
    file_path = os.path.join("pdfs", file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_pdf(file_path):
    """Load PDF using PDFPlumberLoader"""
    try:
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def split_text(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

# Retriever building functions
def build_semantic_retriever(documents, embedding_model, k=4):
    """Build semantic retriever using FAISS"""
    texts = [doc.page_content for doc in documents]
    metadata = [doc.metadata for doc in documents]
    try:
        vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadata)
        return vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        st.error(f"Error building semantic retriever: {e}")
        return None

def build_bm25_retriever(documents, k=4):
    """Build BM25 retriever"""
    try:
        return BM25Retriever.from_documents(documents, preprocess_func=word_tokenize, k=k)
    except Exception as e:
        st.error(f"Error building BM25 retriever: {e}")
        return None

def build_ensemble_retriever(semantic_retriever, bm25_retriever, weights=None):
    """Build ensemble retriever that combines semantic and keyword search"""
    if weights is None:
        weights = [0.5, 0.5]
    try:
        return EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=weights
        )
    except Exception as e:
        st.error(f"Error building ensemble retriever: {e}")
        return None

# Advanced context processing
def rerank_documents(question, documents, embedding_model, top_k=4):
    """Rerank documents using cross-attention relevance"""
    if not documents:
        return []
    
    # Get embeddings for question and documents
    try:
        question_embedding = embedding_model.embed_query(question)
        doc_embeddings = [embedding_model.embed_query(doc.page_content) for doc in documents]
        
        # Calculate similarity scores
        similarities = cosine_similarity([question_embedding], doc_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return reranked documents
        return [documents[i] for i in top_indices]
    except Exception as e:
        st.error(f"Error reranking documents: {e}")
        return documents[:top_k]  # Fall back to first top_k documents

def dynamic_retriever_selection(question, semantic, bm25, ensemble):
    """Dynamically select best retriever based on question type"""
    question_lower = question.lower()
    
    # For definitional questions, prefer semantic search
    if any(term in question_lower for term in ["define", "explain", "what is", "describe"]):
        return semantic, "Semantic"
    
    # For specific fact questions, prefer BM25
    elif any(term in question_lower for term in ["when", "where", "who", "specific", "exact"]):
        return bm25, "BM25"
    
    # Default to ensemble for balanced retrieval
    else:
        return ensemble, "Ensemble"

# Answer generation
def answer_question(question, documents, model, max_length=2048):
    """Generate answer using selected LLM"""
    if not documents:
        return "No relevant documents found to answer the question."
    
    # Format context from documents
    context = "\n\n".join([
        f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}): {doc.page_content}" 
        for i, doc in enumerate(documents)
    ])
    
    # Create prompt
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
        st.error(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}"

# Feedback collection and processing
def collect_user_feedback(question, answer, documents, retriever_type):
    """Collect and store user feedback"""
    with st.expander("Provide feedback on this answer"):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            quality = st.select_slider(
                "Rate answer quality:",
                options=["Poor", "Fair", "Good", "Excellent"],
                value="Good"
            )
            
            relevance = st.select_slider(
                "Rate document relevance:",
                options=["Poor", "Fair", "Good", "Excellent"],
                value="Good"
            )
        
        with col2:
            better_answer = st.text_area("Suggest a better answer (optional):")
            specific_feedback = st.text_input("Specific feedback (optional):")
        
        if st.button("Submit Feedback"):
            # Convert ratings to numerical scores
            quality_score = {"Poor": 0.0, "Fair": 0.33, "Good": 0.67, "Excellent": 1.0}[quality]
            relevance_score = {"Poor": 0.0, "Fair": 0.33, "Good": 0.67, "Excellent": 1.0}[relevance]
            
            # Store feedback
            timestamp = datetime.now().isoformat()
            feedback_data = {
                "question": question,
                "generated_answer": answer,
                "better_answer": better_answer,
                "quality_score": quality_score,
                "relevance_score": relevance_score,
                "retriever_type": retriever_type,
                "specific_feedback": specific_feedback,
                "timestamp": timestamp,
                "document_ids": [doc.metadata.get("source", "Unknown") for doc in documents]
            }
            
            # Add to session state
            feedback_key = f"{question}_{timestamp}"
            st.session_state.feedback_store[feedback_key] = feedback_data
            
            # Save to disk
            with open("feedback/feedback_data.json", "w") as f:
                json.dump(st.session_state.feedback_store, f, indent=2)
            
            st.success("Feedback submitted successfully!")
            
            # Update retriever weights based on feedback
            update_retriever_weights(retriever_type, quality_score)

def update_retriever_weights(retriever_type, quality_score):
    """Dynamically adjust retriever weights based on feedback"""
    # Start with default weights
    if 'retriever_weights' not in st.session_state:
        st.session_state.retriever_weights = [0.5, 0.5]  # [semantic, bm25]
    
    # Adjust weights slightly based on feedback
    weights = st.session_state.retriever_weights
    
    if retriever_type == "Semantic" and quality_score > 0.5:
        # Increase semantic weight
        weights[0] = min(0.8, weights[0] + 0.05)
        weights[1] = 1.0 - weights[0]
    elif retriever_type == "BM25" and quality_score > 0.5:
        # Increase BM25 weight
        weights[1] = min(0.8, weights[1] + 0.05)
        weights[0] = 1.0 - weights[1]
    
    # Update weights in session state
    st.session_state.retriever_weights = weights
    
    # Rebuild ensemble retriever with new weights
    if st.session_state.semantic_retriever and st.session_state.bm25_retriever:
        st.session_state.ensemble_retriever = build_ensemble_retriever(
            st.session_state.semantic_retriever,
            st.session_state.bm25_retriever,
            weights=weights
        )

# Evaluation metrics
def calculate_system_metrics():
    """Calculate system performance metrics based on feedback"""
    if not st.session_state.feedback_store:
        return None
    
    metrics = {
        "total_questions": len(st.session_state.feedback_store),
        "avg_quality": 0.0,
        "avg_relevance": 0.0,
        "retriever_performance": {"Semantic": 0.0, "BM25": 0.0, "Ensemble": 0.0},
        "retriever_counts": {"Semantic": 0, "BM25": 0, "Ensemble": 0}
    }
    
    for feedback_key, feedback in st.session_state.feedback_store.items():
        metrics["avg_quality"] += feedback.get("quality_score", 0)
        metrics["avg_relevance"] += feedback.get("relevance_score", 0)
        
        retriever_type = feedback.get("retriever_type", "Unknown")
        if retriever_type in metrics["retriever_performance"]:
            metrics["retriever_performance"][retriever_type] += feedback.get("quality_score", 0)
            metrics["retriever_counts"][retriever_type] += 1
    
    # Calculate averages
    if metrics["total_questions"] > 0:
        metrics["avg_quality"] /= metrics["total_questions"]
        metrics["avg_relevance"] /= metrics["total_questions"]
    
    # Calculate average per retriever
    for retriever in metrics["retriever_performance"]:
        if metrics["retriever_counts"][retriever] > 0:
            metrics["retriever_performance"][retriever] /= metrics["retriever_counts"][retriever]
    
    return metrics

# Main application UI
def main():
    st.title("Advanced RAG System with Feedback")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_name = st.selectbox(
            "Select LLM model",
            ["mistral:latest", "llama2:latest", "phi:latest"],
            index=0
        )
        
        # Embedding model selection
        embedding_model_name = st.selectbox(
            "Select embedding model",
            ["sentence-transformers/all-mpnet-base-v2"],
            index=0
        )


        
        # Chunking parameters
        chunk_size = st.slider("Chunk size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk overlap", 50, 500, 200)
        
        # Retriever parameters
        k_documents = st.slider("Number of documents to retrieve", 2, 10, 4)
        
        # Document processing button
        if st.session_state.documents is not None:
            if st.button("Reprocess documents"):
                with st.spinner("Processing documents..."):
                    st.session_state.chunked_documents = split_text(
                        st.session_state.documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Load embedding model
                    embedding_model = load_embedding_model(embedding_model_name)
                    
                    # Build retrievers
                    st.session_state.semantic_retriever = build_semantic_retriever(
                        st.session_state.chunked_documents,
                        embedding_model,
                        k=k_documents
                    )
                    
                    st.session_state.bm25_retriever = build_bm25_retriever(
                        st.session_state.chunked_documents,
                        k=k_documents
                    )
                    
                    st.session_state.ensemble_retriever = build_ensemble_retriever(
                        st.session_state.semantic_retriever,
                        st.session_state.bm25_retriever
                    )
                
                st.success("Documents reprocessed!")
        
        # View system metrics
        if st.button("View System Metrics"):
            metrics = calculate_system_metrics()
            if metrics:
                st.write("### System Metrics")
                st.write(f"Total questions: {metrics['total_questions']}")
                st.write(f"Average quality score: {metrics['avg_quality']:.2f}")
                st.write(f"Average relevance score: {metrics['avg_relevance']:.2f}")
                
                st.write("### Retriever Performance")
                for retriever, score in metrics["retriever_performance"].items():
                    count = metrics["retriever_counts"][retriever]
                    if count > 0:
                        st.write(f"{retriever}: {score:.2f} (used {count} times)")
            else:
                st.info("No feedback data available yet.")
        
        # Current weights display
        if 'retriever_weights' in st.session_state:
            st.write("### Current Retriever Weights")
            st.write(f"Semantic: {st.session_state.retriever_weights[0]:.2f}")
            st.write(f"BM25: {st.session_state.retriever_weights[1]:.2f}")
    
    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # PDF upload
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
        
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                # Save and load PDF
                file_path = upload_pdf(uploaded_file)
                st.session_state.documents = load_pdf(file_path)
                
                # Process documents
                st.session_state.chunked_documents = split_text(
                    st.session_state.documents,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Load models
                embedding_model = load_embedding_model(embedding_model_name)
                
                # Build retrievers
                st.session_state.semantic_retriever = build_semantic_retriever(
                    st.session_state.chunked_documents,
                    embedding_model,
                    k=k_documents
                )
                
                st.session_state.bm25_retriever = build_bm25_retriever(
                    st.session_state.chunked_documents,
                    k=k_documents
                )
                
                st.session_state.ensemble_retriever = build_ensemble_retriever(
                    st.session_state.semantic_retriever,
                    st.session_state.bm25_retriever
                )
            
            st.success(f"Processed {len(st.session_state.chunked_documents)} document chunks")
    
    with col2:
        # Chat interface
        st.header("Chat with your documents")
        
        # Show chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Get user question
        if question := st.chat_input("Ask a question about your documents"):
            # Add user question to chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Display user question
            with st.chat_message("user"):
                st.write(question)
            
            # Generate answer
            with st.chat_message("assistant"):
                if st.session_state.chunked_documents and st.session_state.semantic_retriever:
                    with st.spinner("Searching documents..."):
                        # Load model
                        model = load_ollama_model(model_name)
                        
                        # Select retriever
                        selected_retriever, retriever_type = dynamic_retriever_selection(
                            question,
                            st.session_state.semantic_retriever,
                            st.session_state.bm25_retriever,
                            st.session_state.ensemble_retriever
                        )
                        
                        # Retrieve documents
                        retrieved_documents = selected_retriever.invoke(question)
                        
                        # Rerank documents
                        embedding_model = load_embedding_model(embedding_model_name)
                        reranked_documents = rerank_documents(
                            question,
                            retrieved_documents,
                            embedding_model,
                            top_k=k_documents
                        )
                        
                        # Generate answer
                        answer = answer_question(question, reranked_documents, model)
                        
                        # Display answer
                        st.write(answer)
                        
                        # Add answer to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                        # Show documents used
                        with st.expander("Documents used"):
                            for i, doc in enumerate(reranked_documents):
                                st.markdown(f"**Document {i+1}** (Source: {doc.metadata.get('source', 'Unknown')})")
                                st.write(doc.page_content)
                                st.divider()
                        
                        # Collect feedback
                        collect_user_feedback(question, answer, reranked_documents, retriever_type)
                else:
                    st.error("Please upload a PDF document first.")

if __name__ == "__main__":
    main()