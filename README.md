# RAG Agent - Multi-Document PDF Q&A

## Overview

This project is a Retrieval-Augmented Generation (RAG) application that
allows users to upload PDF documents and ask questions grounded strictly
in their content.

It combines:

-   Streamlit for the user interface
-   LangChain (LCEL) for orchestration
-   FAISS for vector similarity search
-   Groq LLM for fast inference
-   Multiple embedding providers (HuggingFace Local, HuggingFace API,
    OpenAI)

The system retrieves relevant document chunks and generates answers
using only the retrieved context, with source attribution.

------------------------------------------------------------------------

## Key Features

-   Upload and index multiple PDFs
-   Configurable chunking and retrieval settings
-   Multiple embedding provider options
-   Groq-powered LLM responses

------------------------------------------------------------------------

## Architecture

High-level flow:

![Architecture](https://github.com/N-Vasu-Reddy/Doc_Chat/blob/main/assets/rag_architecture.png)

------------------------------------------------------------------------

## Installation

### 1. Clone the Repository

git clone `<repo-url>`

cd `<repo-name>

### 2. Create Virtual Environment (Recommended)
```
python -m venv venv

```
```
source venv/bin/activate \# macOS/Linux
venv\Scripts\activate \# Windows
```


### 3. Install Dependencies

Core dependencies:
```
pip install streamlit langchain-community langchain-groq
langchain-text-splitters langchain-core faiss-cpu pypdf
```
Optional (depending on embedding provider):

For HuggingFace local: pip install langchain-huggingface
sentence-transformers

For OpenAI embeddings: pip install langchain-openai

------------------------------------------------------------------------

## Running the Application
```
streamlit run app.py
```

Open the local URL displayed in your terminal.

------------------------------------------------------------------------

## How to Use

1.  Enter your Groq API key in the sidebar.
2.  Choose your embedding provider.
3.  Upload one or more PDF files.
4.  Click Index Documents.
5.  Ask questions in the chat box.

The system will: 
- Retrieve relevant document chunks.
- Generate a grounded response.
- Display the source files used.

------------------------------------------------------------------------

## Configuration Options

From the sidebar, you can adjust:

-   Embedding provider and model
-   Chunk size and overlap
-   Top-K retrieval count
-   LLM temperature

------------------------------------------------------------------------

## Notes

-   The FAISS index is stored in memory (cleared on restart).
-   If you change embedding providers, clear the index first.
-   Answers are restricted to uploaded documents.


