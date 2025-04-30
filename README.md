# LIBRAG: Retrieval-Augmented Search for Boston Public Library

**LIBRAG** is a Retrieval-Augmented Generation (RAG) system designed to improve access to the Boston Public Library’s (BPL) Digital Commonwealth archive. It enables users to ask natural language questions and retrieve highly relevant documents, complete with metadata, images, and audio entries.

---

## Project Overview

The Boston Public Library needed a better way for users to explore their extensive digital collection through intuitive search methods rather than traditional keyword queries.  
LIBRAG addresses this need by using a RAG pipeline that retrieves documents semantically and generates informative, source-backed responses.

Key enhancements to the project include:

- Full ingestion and embedding of over 1.2 million metadata entries.
- Multi-field embedding (title, abstract, creator, date, and combined fields).
- BM25 reranking of search results to improve answer relevance.
- Support for retrieving and displaying images and audio metadata in search results.
- A user-friendly Streamlit UI for seamless interaction.

---

## Live Demo

Try the hosted demo here:  
👉 [Hugging Face Spaces - BPL-RAG-Spring-2025](https://huggingface.co/spaces/spark-ds549/BPL-RAG-Spring-2025)

---

## System Architecture

The system is organized into two main pipelines: **Data Preparation** and **Query Handling**.

### Data Preparation Pipeline

- Scrape metadata from the Digital Commonwealth API.
- Preprocess fields (title, abstract, creator, date).
- Combine key fields into a single text block for better embedding.
- Embed records using the `all-MiniLM-L6-v2` model from Hugging Face.
- Store embeddings in a Pinecone vector database.

### Query Handling Pipeline

- Accept user natural language query through Streamlit UI.
- Preprocess and embed the query.
- Retrieve top matching documents from Pinecone.
- Rerank retrieved documents using BM25 based on metadata relevance.
- Generate a final response using GPT-4o-mini and present retrieved sources, including image previews and grouped audio metadata when applicable.

---

## Features

- **Large-scale Embedding**: Successfully processed over 1.2 million items from BPL’s archive.
- **Multi-Field Retrieval**: Embedding includes titles, abstracts, creators, and dates for richer semantic search.
- **BM25 Reranking**: Reorders retrieved documents to prioritize field matches (e.g., better date or title alignment).
- **Image and Audio Integration**:
  - Images are retrieved and displayed alongside text-based results.
  - Audio metadata records are reconstructed from grouped fragments for full context.
- **Cost Optimization**: Open-source embeddings and GPT-4o-mini model are used to minimize API costs while maintaining quality.
- **Hosting**: Deployed publicly through Hugging Face Spaces for accessibility.

---

## Setup Instructions

### Prerequisites

- Python 3.12.4 or later
- Pinecone account for vector storage
- OpenAI account for LLM API access (or alternative model integration)

### Installation

Clone the repository and set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your `.env` file:

```plaintext
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
```

---

## Loading and Embedding Metadata

### Scrape Metadata

Run the following to download metadata:

```bash
python load_scraper.py <BEGIN_PAGE> <END_PAGE>
```

This will save a `.json` file of metadata entries for the specified page range.

### Embed Metadata into Pinecone

Run the following to upload the embeddings:

```bash
python load_pinecone.py <BEGIN_INDEX> <END_INDEX> <PATH_TO_JSON> <PINECONE_INDEX_NAME>
```

Make sure your Pinecone index is created beforehand with the correct vector dimension (384 for MiniLM-L6-v2).

---

## Running the Application

Update `streamlit_app.py` with your Pinecone index name:

```python
INDEX_NAME = "your-pinecone-index-name"
```

Launch the application:

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501/`.  
Response times may vary depending on dataset size and retrieval/re-ranking operations.

---

## Current Limitations

- **Speed**: Retrieval and metadata enrichment can take 25–70 seconds due to sequential Digital Commonwealth API calls.
- **Metadata Bottleneck**: After retrieving vector matches, full metadata is fetched live for reranking, adding delay.
- **Scaling Costs**: Pinecone costs scale with the volume of embedded vectors; full ingestion of the archive is costly.

---

## Future Improvements

- Build a pre-cached local metadata database to eliminate live API calls.
- Add native audio playback support in Streamlit using `st.audio()`.
- Enhance query classification to prioritize image or audio results when contextually appropriate.
- Extend metadata grouping logic for additional media types (e.g., manuscripts, videos).
- Improve retrieval robustness by refining prompt engineering and query parsing.

---

## License

This project is licensed under the [MIT License](LICENSE).
