# Chat with PDF using `hybrid` RAG
A simple RAG (Retrieval-Augmented Generation) system using Deepseek, LangChain, and Streamlit to chat with PDFs and answer complex questions about your local documents. This project improves the accuracy of the RAG by using a hybrid approach of retrieval leveraging both **semantic** and **bm25** Best match 25 search.....vector_store.as_retriever() allows the retrieval of relevant documents based on semantic similarity.


# Pre-requisites

Activate the Virtual Environment
```
source my_env1/bin/activate
.\my_env1\Scripts\Activate

```

Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the Deepseek model:

```bash
ollama pull deepseek-r1:14b
ollama list
ollama serve
```

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

# Run
Run the Streamlit app:

```bash
streamlit run hybrid_pdf_rag.py
```
