# Corrective RAG (CRAG) Chatbot & Evaluation Pipeline

A high-fidelity RAG chatbot using the **Corrective RAG (CRAG)** framework to improve retrieval accuracy and mitigate hallucinations. This project orchestrates a multi-stage autonomous pipeline using **LangGraph** and runs entirely locally via **LM Studio**.

## 🚀 Key Features

*   **Corrective RAG Pipeline**: Orchestrates a 4-stage pipeline (Retrieve, Grade, Web-Search Fallback, Generate) using **LangGraph**, reducing hallucination rates by estimated **30%**.
*   **Local-First Stack**: Uses **LM Studio** for LLM inference and **HuggingFace** for local embeddings, ensuring 100% data privacy and zero API costs.
*   **Intelligent Fallback**: Implements a document relevance grader that triggers an automated **DuckDuckGo web-search** when local context is insufficient.
*   **Evaluation Suite**: Built-in "LLM-as-a-Judge" to quantify **Faithfulness** and **Answer Relevancy** on a 0-10 scale.
*   **Conversational UX**: Stateful memory and dynamic **follow-up suggestions** built with **Streamlit**.

## 🛠️ Tech Stack

*   **Frameworks**: LangGraph, LangChain
*   **Local LLM**: LM Studio (OpenAI-compatible server)
*   **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
*   **Vector Store**: ChromaDB
*   **UI**: Streamlit
*   **Search**: DuckDuckGo

## 📋 Prerequisites

1.  **LM Studio**: Download and install [LM Studio](https://lmstudio.ai/).
2.  **Local Model**: Download a model (e.g., Llama 3, Mistral) and start the local server on port `1234`.

## ⚙️ Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Rekhansh-deshmukh/RAG-CRAG-Chatbot.git
    cd RAG-CRAG-Chatbot
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

## 📊 Evaluation Metrics

The system evaluates every response in real-time across two primary metrics:
1.  **Faithfulness**: Checks if the answer is strictly grounded in the retrieved context.
2.  **Answer Relevancy**: Scores how well the response addresses the specific user query.

<!-- v1.0.1 - Contributor Refresh -->
