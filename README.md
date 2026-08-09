# Corrective RAG System

A Retrieval-Augmented Generation system that goes one step further than standard RAG by **checking the relevance of retrieved documents before answering**, and automatically **rewriting the question and re-searching** if the retrieved content turns out to be irrelevant.

---

## Overview

Standard RAG systems retrieve chunks and generate an answer directly, even if the retrieved content isn't actually relevant to the question. This project implements a **Corrective RAG** pipeline: after retrieving document chunks for a question, an LLM "grader" evaluates whether each chunk is actually relevant. If none of the retrieved chunks are relevant, the system automatically rewrites the question to improve retrieval and searches again — only then generating the final answer, strictly grounded in the relevant context, along with its sources.

---

## What the Notebook Does

1. **Data Loading** — loads PDF course/reference materials.
2. **EDA** — explores the loaded documents: number of pages per file, and statistics/distribution of page text lengths.
3. **Preprocessing** — removes empty pages and cleans extracted text (removing extra spaces and blank lines).
4. **Chunking** — splits the cleaned documents into overlapping chunks and analyzes the resulting chunk length statistics/distribution.
5. **Embeddings & Vector Database** — embeds the chunks with a sentence-transformer model and stores them in a FAISS vector database.
6. **Retriever** — builds a similarity-based retriever and tests it with a sample question.
7. **Relevance Grading** — uses an LLM with a structured output schema to grade whether each retrieved chunk is actually relevant to the question (yes/no).
8. **Corrective Retrieval** — if none of the retrieved chunks are graded relevant, the question is automatically rewritten by the LLM to improve retrieval, and the search is repeated with the new query.
9. **Answer Generation** — generates a final answer strictly from the relevant context (or a "not found" message if nothing relevant exists), along with the source documents used.

---

## Tech Stack

- LangChain / LangChain Community / LangChain Hugging Face — RAG pipeline components
- Sentence-Transformers (`all-MiniLM-L6-v2`) — text embeddings
- FAISS — vector database for similarity search
- Groq (LLaMA 3.1 8B Instant) — LLM used for grading, query rewriting, and answer generation
- Pydantic — structured output schema for the relevance grader
- PyPDF — PDF loading
- Pandas / Matplotlib / Seaborn — data exploration and visualization

---

## How to Use

1. Add your Groq API key.
2. Place the PDF materials in the dataset path used by the notebook.
3. Run the notebook top to bottom: it will load and clean the documents, chunk them, build the vector database, and set up the retriever, grader, and answer chain.
4. Ask a question. The system will:
   - Retrieve candidate chunks
   - Grade each chunk's relevance
   - Rewrite and re-search automatically if nothing relevant was found
   - Generate a final grounded answer along with its sources

---

## Possible Improvements

- Add a maximum number of corrective retries instead of a single rewrite-and-retry step
- Log/evaluate how often the corrective (rewrite) path is triggered to measure retrieval quality
- Experiment with different embedding models or grading prompts for better accuracy
