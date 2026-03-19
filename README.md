# RAG Chatbot using Groq API

## Overview

An intelligent chatbot built using Retrieval-Augmented Generation (RAG) and the Groq API. It retrieves relevant information from a vector database and generates accurate, context-aware responses.

---

## Features

* Context-aware question answering
* Reduced hallucinations using retrieved data
* Fast inference with Groq API
* Scalable with vector databases (e.g., Pinecone)

---

## Tech Stack

* Python
* Groq API
* Vector Database (Pinecone or similar)
* LangChain / Custom RAG pipeline

---

## How It Works

1. User submits a query through the interface (index.html)
2. Backend (rag_server.py) processes the query
3. Relevant documents are retrieved from the vector database
4. Retrieved context is sent to the LLM via Groq API
5. The chatbot generates a grounded response

---

## Project Structure

```
├── env/                # Virtual environment
├── .env               # API keys and environment variables
├── .gitignore         # Ignored files
├── index.html         # Frontend interface
├── rag_server.py      # Backend RAG pipeline
├── requirements.txt   # Dependencies
```

---

## Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---

## Future Improvements

* Add conversation memory
* Improve retrieval accuracy
* Deploy as a web application

---

## License

This project is for educational purposes.

---

## Acknowledgements

* Groq API
* Open-source RAG ecosystem
