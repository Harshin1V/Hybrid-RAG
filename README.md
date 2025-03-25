# Advanced RAG System with Feedback Integration

The **Advanced RAG System** is a Retrieval-Augmented Generation (RAG) application built using Streamlit. It allows users to upload PDF documents, ask questions, and receive accurate answers using a combination of semantic and BM25 retrievers. Additionally, the system incorporates a feedback loop to enhance the accuracy of responses by adjusting retriever weights based on user feedback.

## Features
- **Document Processing**: Efficiently processes PDF documents using chunking.
- **Hybrid Retrieval**: Combines semantic and BM25 retrieval methods using a weighted ensemble.
- **RAG Pipeline**: Generates answers using an LLM (e.g., Mistral, LLaMA).
- **Feedback Mechanism**: Collects user feedback and dynamically adjusts retriever weights.
- **Feedback Dashboard**: Displays collected feedback for transparency.

## Project Structure
```
├── pdfs                 # Uploaded PDF documents
├── models               # Model storage (if applicable)
├── feedback             # Feedback data in JSON format
├── hybrid_rag.py        # Main application script
├── requirements.txt     # Dependencies
└── README.md            # Project Documentation
```

## Installation
1. **Clone the Repository:**
    ```bash
    git clone <repo-url>
    cd Hybrid-RAG-main
    ```
2. **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv my_env
    source my_env/bin/activate
    ```
3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Download NLTK Data:**
    ```bash
    python -c "import nltk; nltk.download('punkt')"
    ```

## Usage
1. **Run the Application:**
    ```bash
    streamlit run hybrid_rag.py
    ```
2. **Upload a PDF:** Use the file uploader to add a PDF document.
3. **Ask a Question:** Enter your question in the provided text box.
4. **Provide Feedback:** Mark the answer as helpful or not to enhance the system.

## Configuration
- **Model Selection:** Choose from Mistral, LLaMA, or Phi for LLM generation.
- **Chunk Size & Overlap:** Adjust the document chunking parameters.
- **Feedback Dashboard:** View detailed feedback data.

## Feedback Mechanism
- The system collects feedback (Yes/No) after each answer.
- Feedback data is stored in `feedback/feedback_data.json`.
- The retriever weights are adjusted using the following formula:
    ```python
    positive_ratio = positive_feedback / total_feedback
    st.session_state.retriever_weights = [positive_ratio, 1 - positive_ratio]
    ```

## Troubleshooting
- **Model Loading Error:** Ensure models like Mistral or LLaMA are correctly installed.
- **PDF Processing Issues:** Confirm PDF files are not corrupted.
- **Feedback Dashboard Not Displaying:** Ensure feedback data is stored in `feedback_data.json`.

## Contributing
If you'd like to contribute to this project, feel free to submit a pull request or report issues.

## License
This project is licensed under the MIT License.

---

**Happy Exploring with Advanced RAG!**

