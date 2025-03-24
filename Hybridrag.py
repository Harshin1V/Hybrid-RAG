import streamlit as st
# import ray
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
import nltk
import json
from peft import get_peft_model, LoraConfig
from transformers import AutoModel,AutoModelForCausalLM, Trainer, TrainingArguments
from trl import PPOTrainer
nltk.download('punkt')
from nltk.tokenize import word_tokenize


# Load fine-tuned Mistral-7B-Instruct or Phi-2 model for cost efficiency
# model = OllamaLLM(model="mistral-7b-instruct")
model = OllamaLLM(model="mistral:latest")

# Choose embedding model: SBERT, OpenAI, or ColBERT
embedding_model = SentenceTransformer('all-mpnet-base-v2', trust_remote_code=True)
# colbert_model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
# colbert_tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")

# RLHF Feedback Store
feedback_store = {}

# # Ray for distributed processing
# ray.init(ignore_reinit_error=True)

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = 'pdfs/'

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(documents)

def fine_tune_model():
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct")
    config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1)
    peft_model = get_peft_model(model, config)
    training_args = TrainingArguments(output_dir="./mistral_finetuned", per_device_train_batch_size=2, num_train_epochs=3)
    trainer = Trainer(model=peft_model, args=training_args)
    trainer.train()

def build_semantic_retriever(documents):
    texts = [doc.page_content for doc in documents]
    
    # Use HuggingFaceEmbeddings wrapper
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    
    # FAISS expects an embedding function from LangChain, not SentenceTransformer directly
    vectorstore = FAISS.from_texts(texts, embedding_model)
    
    return vectorstore.as_retriever()

# def build_colbert_retriever(texts):
#     embedding_model = HuggingFaceEmbeddings(model_name="facebook/colbertv2")  # Ensure this model exists
#     vectorstore = FAISS.from_texts(texts, embedding_model)
#     return vectorstore.as_retriever()

def build_bm25_retriever(documents):
    return BM25Retriever.from_documents(documents, preprocess_func=word_tokenize)

def dynamic_retriever_selection(question, semantic, bm25):
    if "define" in question.lower() or "explain" in question.lower():
        return semantic
    else:
        return bm25

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

def collect_user_feedback(question, answer):
    feedback = st.radio(f"Was this answer helpful for: '{question}'?", ["Yes", "No"])
    if feedback == "No":
        better_answer = st.text_input("Provide a better answer (Optional):")
        feedback_store[question] = {"answer": answer, "user_feedback": better_answer}
        with open("feedback_data.json", "w") as f:
            json.dump(feedback_store, f)

def fine_tune_with_feedback():
    feedback_data = json.load(open("feedback_data.json"))
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct")
    trainer = PPOTrainer(model)
    for question, feedback in feedback_data.items():
        trainer.train_on_feedback(question, feedback["user_feedback"])
    model.save_pretrained("./mistral_finetuned")

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)

    semantic_retriever = build_semantic_retriever(chunked_documents)
    bm25_retriever = build_bm25_retriever(chunked_documents)
    # colbert_retriever = build_colbert_retriever(chunked_documents)
    
    question = st.chat_input()
    if question:
        st.chat_message("user").write(question)
        selected_retriever = dynamic_retriever_selection(question, semantic_retriever, bm25_retriever)
        related_documents = selected_retriever.invoke(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)

        # Collect RLHF feedback
        collect_user_feedback(question, answer)

# Deploy to Hugging Face Spaces compatibility ensured
