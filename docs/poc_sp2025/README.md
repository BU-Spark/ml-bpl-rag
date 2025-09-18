# LIBRAG: A Retrieval-Augmented Generation System for Boston Public Library

This project presents a working Proof-of-Concept ipeline that enables semantic, multimodal search across historical metadata from the Boston Public Library, powered by the Digital Commonwealth API. Built using LangChain, Pinecone, HuggingFace, and OpenAI, our system is designed to be fast, extensible, and fully reproducible — capable of responding to natural language queries using structured archival data, images, and audio.

## Overview

We implemented a Retrieval-Augmented Generation architecture for public library metadata search. Our pipeline enables users to:

- Search BPL metadata semantically using a natural language query.
- Retrieve and rerank relevant results using vector similarity and BM25.
- Generate librarian-style summaries with GPT-4o-mini.
- Reference exact source fields for transparency.
- Extend the system to support image and audio metadata using embedding-based similarity.

## Technologies Used

- Python 3.10 (Jupyter Notebook)
- Pinecone (Vector DB)
- HuggingFace Transformers (MiniLM model)
- LangChain (RAG orchestration)
- OpenAI GPT-4o-mini (answer generation)
- Google Cloud SQL (metadata caching)
- Rank_BM25 (re-ranking)
- PIL, Tesseract OCR (image-to-text embedding)
- Whisper ASR (audio-to-text embedding)

## Notebook Workflow

### Step 1: Environment Setup

All dependencies are installed using pip with user-level permissions. SCC-compatible binary paths are configured, and environment variables are loaded from `.env`.

### Step 2: Embedding Initialization

We use `sentence-transformers/all-MiniLM-L6-v2` via LangChain’s HuggingFaceEmbeddings wrapper to convert text into vector representations.

### Step 3: Metadata Collection

Rather than reading from static files, we implemented a live scraping function that calls the Digital Commonwealth API, paginates through results, and appends metadata from each page into memory. This avoids rate limits and unnecessary I/O.

### Step 4: Preprocessing

Relevant metadata fields (titles, abstracts, notes) are chunked using LangChain's RecursiveCharacterTextSplitter. The resulting segments are converted into LangChain `Document` objects with source metadata.

### Step 5: Vector Indexing with Pinecone

Documents are embedded and stored in a Pinecone index using LangChain’s PineconeVectorStore. The `index.add_documents()` method efficiently uploads both content and deterministic UUIDs.

### Step 6: Semantic Retrieval

We issue a natural language query and use Pinecone’s similarity search to retrieve top-k matches. Each result is previewed with its metadata field and truncated content.

### Step 7: BM25 Re-Ranking

To improve result quality, we re-rank the Pinecone results using BM25. For each unique source ID, we fetch full metadata via the BPL API, extract additional fields, and pass them to LangChain's `BM25Retriever`. The re-ranked list often surfaces more informative items for generation.

### Step 8: GPT-4o-mini Generation

Using LangChain’s `PromptTemplate`, we engineer a librarian-style prompt. Each result is labeled with `[Source: ID | Field]`, and the LLM is instructed to reason about the context, validate the information's sufficiency, and return a clean summary inside XML-style tags. We parse and print the final answer.

### Step 9: Image and Audio Metadata Embedding (Extension)

We include standalone scripts to:
- Convert image files to text using Tesseract OCR and embed them into Pinecone as searchable documents.
- Convert audio files (e.g., oral histories) into text using Whisper ASR and embed those results.

These modalities can be retrieved and ranked just like text metadata, supporting multimodal retrieval.

### Step 10: Metadata Caching via Google Cloud SQL

To avoid re-hitting the Digital Commonwealth API for every query, we preloaded all structured metadata into a Google Cloud SQL relational table. This reduced per-query API time from 2 minutes to under 10 seconds. Our pipeline checks the table first before making external requests.

## Reproducibility

- All code is structured into Jupyter Notebook cells with clear markdown explanations.
- Setup, embedding, scraping, reranking, and generation are all modular and parameterized.
- The system can be extended to include images, audio, or PDFs.

## Next Steps

- Extend cloud SQL to store embeddings or pre-ranked suggestions.
- Add user feedback logging to refine re-ranking weights.
- Allow user-controlled filters (e.g., year, genre) and result citations.
- Add audio/image conversion fully automated into the pipeline