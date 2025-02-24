# Research and Problem Understanding (BPL - Spring 2025)
**Data combined from PDFs in this respective folder**

## 1. Problem Statement  

**Goal:** Build a search engine for BPL using RAG.  
**Purpose:** Make digital resources more accessible with natural language queries.  
**Solution:** Use a RAG pipeline to retrieve documents based on metadata + text, then generate AI responses.  

**Challenges:**  
- Metadata weighting is missing, meaning long abstracts often outweigh title matches.  
- BM25 reranking is applied post-retrieval but does not emphasize certain metadata fields.  
- High retrieval latency (~5 minutes) due to inefficient metadata fetching and query processing.  
- Pinecone indexing costs are high, requiring alternatives for storage and retrieval optimization.  

---

## 2. Research Process  

### 2.1 Database & Dataset Details  
**Source:** Digital Commonwealth database.  

**Size:**  
- 1.3M items (text, video, audio, images).  
- 147K full-text OCR documents.  
- Metadata JSON files (up to 135 fields per document).  
- 600K+ still images, fewer than 100K non-text files.  

**Data Processing:**  
- Data is pulled from the Digital Commonwealth API.  
- Metadata extracted using `load_script.py`, outputting JSON files.  
- Chunking is used for large documents to improve search granularity.  

### 2.2 Academic Papers Reviewed  
We examined four key papers related to vector search, metadata embedding, and retrieval-augmented generation (RAG):

- **[Efficient Retrieval-Augmented Generation](https://arxiv.org/abs/2205.09780)** – Discusses optimizing retrieval before passing data to LLMs for faster inference.  
- **[BM25-FIC: Information Content-based Field Weighting for BM25F](https://ceur-ws.org/Vol-2741/paper-11.pdf)** – Proposes an optimized BM25 weighting approach for metadata-heavy retrieval.  
- **[Optimizing BM25 for Hybrid Search](https://arxiv.org/abs/1905.09625)** – Covers BM25 ranking improvements when combining keyword-based and vector-based retrieval.  
- **[Vector Search in a Nutshell](https://towardsdatascience.com/vector-search-in-a-nutshell-6f0a44a5ddaa)** – Explains ANN techniques like FAISS vs. Pinecone vs. Weaviate for cost-effective vector search.  

### 2.3 Open Source Projects Explored  
We evaluated open-source projects for potential integration:

- **[Haystack (Deepset)](https://github.com/deepset-ai/haystack)** – Implements hybrid BM25 + vector search with metadata filters.  
- **[FAISS (Facebook AI)](https://github.com/facebookresearch/faiss)** – A self-hosted alternative to Pinecone for faster, cost-effective vector search.  
- **[Llama 3 (Meta)](https://github.com/meta-llama/llama3)** – A lightweight, open-source LLM designed for low-latency inference and cost-efficient deployment. Evaluated as an alternative to GPT-4o to reduce LLM API costs.  

---

## 3. Research Findings  

### 3.1 Embedding Process  
- Uses all-MiniLM-L6-v2 (Hugging Face model).  
- Vector size: 384 dimensions.  
- Stored in Pinecone vector store.  
- Only key metadata fields embedded due to storage limits.  

### 3.2 Metadata Embedding Approaches  
We evaluated three methods:  

1. **Concatenation Approach**  
   - Combines metadata (title, author, date, abstract, etc.) into a single text string before embedding.  
   - Simple but dilutes metadata signals.  
2. **Separate Metadata Vectors**  
   - Creates distinct embeddings for title, date, abstract, and author.  
   - Granular control, but increases retrieval complexity.  
3. **Hybrid Retrieval (BM25 + Embeddings)**  
   - Uses BM25 for structured fields (titles, dates, authors) and vector search for abstracts.  
   - Best balance of efficiency and accuracy.  

### 3.3 RAG Query Processing  
- User query is converted into embeddings.  
- Compared against stored document vectors.  

### 3.4 Document Retrieval  
- Uses cosine similarity to retrieve the top 100 closest documents.  
- Metadata is fetched from API (slow process).  

### 3.5 Reranking & Content Integration  
- Uses BM25 for metadata-based reranking.  
- Example: If a query mentions "1919," prioritizes metadata with matching date fields.  
- Improves response quality.  

### 3.6 Response Generation  
- Sends retrieved text to GPT-4o-mini.  
- Uses structured prompts to avoid hallucination.  
- UI shows:  
  - **Generated response**.  
  - **Retrieved snippets**.  
  - **Links to source documents**.  

### 3.7 Optimizing Vector Search Performance  
**Issue:** Queries take ~5 minutes due to metadata API delays.  
**Proposed Fixes:**  
- **Preload metadata** into a structured SQL/NoSQL database instead of calling the API per query.  
- **Parallelize metadata retrieval** using synchronous calls.  
- **Implement multi-vector search** (separate indexes for title, date, abstract).  

### 3.8 Alternative Vector Stores to Pinecone  
**Problem:** Pinecone’s high storage costs slow down large dataset queries.  
**Potential Alternatives:**  
- **FAISS:** Local vector search with HNSW indexing (faster & cheaper).  
- **Weaviate:** Open-source hybrid search with built-in metadata filtering.  
- **Milvus:** Scalable ANN search with better self-hosting options.  

---

## 4. Proposed Implementation Plan  

### 4.1 Performance Criteria  
- **Reduce query latency** from ~5 minutes.  
- **Improve ranking** for title & date-based queries.  
- **Lower Pinecone costs** by 50% using FAISS or hybrid retrieval.  

### 4.2 Deliverable Artifacts  
- Prototype word ranking adjustments using BM25 weighting.  
- Performance benchmarks comparing BM25, FAISS, and Pinecone retrieval speeds.  
- Structured prompt templates for LLM queries using metadata.  

---

## 5. References & Resources  

### Academic Papers  
- **[Efficient Retrieval-Augmented Generation](https://arxiv.org/abs/2205.09780)** – Discusses how retrieval optimization reduces LLM overhead for faster inference.  
- **[BM25-FIC: Field Weighting for BM25](https://ceur-ws.org/Vol-2741/paper-11.pdf)** – Proposes BM25 weighting for metadata fields to improve relevance ranking.  
- **[Optimizing BM25 for Hybrid Search](https://arxiv.org/abs/1905.09625)** – Examines BM25 + vector search to refine query precision.  

### Open-Source Tools  
- **[Haystack (BM25 + Vectors)](https://github.com/deepset-ai/haystack)** – Provides BM25 + vector search with metadata filtering.  
- **[FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)** – Self-hosted alternative to Pinecone, supports HNSW indexing for speed.  
- **[Llama 3 (Meta)](https://github.com/meta-llama/llama3)** – Open-source LLM being evaluated as a cost-effective GPT-4o alternative.  
- **[Weaviate Vector Database](https://weaviate.io/)** – Hybrid search with metadata-based filtering.  

### APIs Used  
- **[Digital Commonwealth API](https://www.digitalcommonwealth.org/developers)** – Extracts titles, descriptions, and authors from BPL’s archive.  
- **[Pinecone Documentation](https://docs.pinecone.io/)** – Optimizing vector storage and metadata filtering.  
- **[LangChain Metadata Handling](https://python.langchain.com/docs/modules/data_connection/retrievers/)** – Helps pre-filter metadata before LLM queries.  

---