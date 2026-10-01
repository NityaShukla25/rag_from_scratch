# 🔎 RAG From Scratch

A hands-on implementation of **Retrieval-Augmented Generation (RAG)** built from scratch in Python, without LangChain.

The goal is to understand the internal working of RAG — from document processing and embeddings to semantic retrieval and LLM-based generation.

---

## 🚀 What is RAG?

RAG combines **information retrieval** with **Large Language Models**.

Instead of relying only on the LLM's pretrained knowledge, relevant information is retrieved from an external knowledge base and provided to the LLM as context.

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Representation
    ↓
Similarity Search
    ↓
Top-K Chunks
    ↓
Context
    ↓
LLM
    ↓
Answer
```

---

## 🏗️ Architecture

### Indexing

```text
Documents → Text Extraction → Chunking → Embeddings → Knowledge Base
```

### Query & Generation

```text
User Query
    ↓
Query Embedding
    ↓
Cosine Similarity
    ↓
Top-K Retrieval
    ↓
Context Construction
    ↓
RAG Prompt
    ↓
LLM
    ↓
Final Answer
```

---

## 🧠 Concepts Implemented

- Document loading and text extraction
- Text chunking
- Chunk metadata
- OpenAI embeddings using `text-embedding-3-small`
- Vector representations
- Cosine similarity
- Semantic search
- Top-K retrieval
- Context construction
- RAG prompt construction
- Grounded LLM generation

---

## 📂 Project Structure

```text
rag_from_scratch/
│
├── documents/
│   ├── placement.txt
│   ├── academic.txt
│   └── hostel.txt
│
├── main.py
├── .env
├── .gitignore
└── venv/
```

---

## 🛠️ Tech Stack

- Python
- OpenAI API
- NumPy
- python-dotenv
- Embeddings
- Cosine Similarity

**No LangChain is used in this version.**

---

## ⚙️ Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install openai python-dotenv numpy
```

### 3. Add API key

Create `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

Make sure `.env` is included in `.gitignore`.

### 4. Run

```bash
python main.py
```

---

## 🧪 Example

**Query:**

```text
What CGPA is required for campus placement?
```

**Retrieved Context:**

```text
Students with a minimum CGPA of 7.0 are eligible
to participate in campus placement.
```

**Generated Answer:**

```text
A minimum CGPA of 7.0 is required to participate
in campus placement.
```

---

## ⚠️ Current Limitations

This is intentionally a simple implementation for understanding RAG fundamentals.

Current limitations:

- In-memory vector storage
- Basic paragraph-based chunking
- Brute-force similarity search
- No reranking or hybrid search
- Basic retrieval evaluation

---

## 🚀 Next Steps

The next phase will rebuild this system using **LangChain** and map each LangChain component to the concepts implemented manually.

```text
RAG From Scratch
       ↓
LangChain
       ↓
Advanced RAG
       ↓
Production-style RAG
```
