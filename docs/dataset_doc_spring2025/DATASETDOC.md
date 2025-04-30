## Project Overview

### What is the project name?
LibRAG - Boston Public Library

---

### What is the link to your project’s GitHub repository?
[https://github.com/BU-Spark/ml-bpl-rag/tree/dev](https://github.com/BU-Spark/ml-bpl-rag/tree/dev)

---

### What is the link to your project’s Google Drive folder?
**Note:** This should be a Spark! owned Google Drive folder.  
[https://drive.google.com/drive/folders/1X0ngdjpJyWnabFjFd0gONESmX_Zomqg1?usp=drive_link](https://drive.google.com/drive/folders/1X0ngdjpJyWnabFjFd0gONESmX_Zomqg1?usp=drive_link)

---

### In your own words, what is this project about? What is the goal of this project?

This project, LIBRAG, is about building a smarter way to search the Boston Public Library’s massive digital archive using AI.  
Instead of relying on keyword search, which can be limiting or confusing, the goal is to let users ask natural language questions and get back relevant documents, images, or audio from the collection — along with helpful summaries and direct source links.  
It's designed to make historical information more accessible, accurate, and user-friendly for researchers, students, and the public.

---

### Who is the client for the project?
Boston Public Library

---

### Who are the client contacts for the project?
Eben English

---

### What class was this project part of?
Spark! Machine Learning X-Lab Practicum (CS549)

## ***Dataset Information***

### What data sets did you use in your project?

We used metadata records from the Digital Commonwealth API, which provides access to digitized materials from libraries, museums, and historical societies across Massachusetts. The data includes structured metadata for over 1.2 million items, including images, audio recordings, text files, and more.

**API Endpoint (for reference):**  
https://www.digitalcommonwealth.org/search.json

---

### Please provide a link to any data dictionaries for the datasets in this project.

Below is a sample data dictionary representing key fields we used from the metadata.

| Field Name                | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `id`                     | Unique item identifier (e.g., `commonwealth:4b29gx20r`)                     |
| `title_info_primary_tsi` | Title of the item                                                           |
| `abstract_tsi`           | Short description or abstract                                               |
| `creator_tsi`            | Author or creator name                                                      |
| `date_tsim`              | Associated date or publication date                                         |
| `format`                 | Media format (e.g., image, audio, text)                                     |
| `subject_tsim`           | Subject tags or keywords                                                    |
| `identifier_uri`         | Direct URL to the item on Digital Commonwealth                              |
| `type`                   | Type of digital object (e.g., Still Image, Sound Recording, Manuscript)     |

> This is a selected subset of metadata fields used. The full dataset includes up to 135 fields.

---

### What keywords or tags would you attach to the data set?

- Retrieval-Augmented Generation (RAG)  
- Digital Libraries  
- Metadata Search  
- Natural Language Processing  
- Image Retrieval  
- Audio Metadata  
- Public Archives  
- Civic Tech  
- Summarization  
- Embeddings  
- BM25  
- Pinecone  
- Boston Public Library  
- Streamlit

---

### Domain(s) of Application

- **Natural Language Processing (NLP)**  
  - Semantic Search  
  - Summarization  
  - Named Entity Recognition (future extension)

- **Computer Vision (metadata-level only)**  
  - Image Retrieval via metadata  
  - Image Preview Display  

- **Civic Tech**  
  - Public Archive Access  
  - Digital Humanities  
  - Library Tech Innovation  

---

### Motivation

**For what purpose was the dataset created?**  
The metadata was created to catalog and preserve access to Massachusetts’s cultural and historical records. However, the default keyword search interface provided by the Digital Commonwealth lacks semantic understanding and contextual relevance.

**Our project’s goal** was to make this data more accessible through natural language queries by building a Retrieval-Augmented Generation (RAG) system that semantically retrieves and ranks the most relevant materials, while supporting modern features like image/audio display and source traceability.

## ***Composition***

### What do the instances that comprise the dataset represent?
The instances represent metadata records associated with digitized historical items from the Digital Commonwealth archive. These include records for still images, audio recordings, text documents, manuscripts, and other digitized formats. Each instance is composed of descriptive metadata, including title, abstract, creator, subject tags, format, and date.

The dataset is **multimodal** in nature, though it only contains metadata:
- **Text**: Descriptions, titles, authors, dates
- **Linked media**: References to image/audio files (via URLs)
- **Tabular**: Structured metadata in JSON format

---

### How many instances are there in total?
- Total metadata records: ~1.2 million
  - Still Images: ~600,000+
  - OCR Text Docs: ~147,000
  - Other (audio, video, manuscripts): < 100,000

---

### Is this dataset complete or a sample?
This is a **partial sample** of the full Digital Commonwealth repository. Due to cost and performance constraints, we embedded a subset (~600k records) for the demo version. The dataset is representative, as we scraped evenly across page ranges to reflect the diversity of formats and collections.

---

### What data does each instance consist of?
Each instance contains **structured metadata**, not the raw media content. Fields include:
- `title`, `abstract`, `creator`, `format`, `date`, `subject`, `uri`
- Some fields may be missing depending on the completeness of the original record

---

### Is there any missing information?
Yes. Some records lack:
- Creator names
- Abstracts or descriptions
- Subject tags

This is due to the variability of metadata completeness from original contributing institutions.

---

### Are there recommended data splits?
No formal splits are used. This dataset is used solely for retrieval-based applications, not training machine learning models.

---

### Are there errors, noise, or redundancies?
Yes. Common issues include:
- Duplicate records across collections
- Misspellings or inconsistently formatted names and subjects
- Missing or malformed dates

These inconsistencies come from upstream data sources and have not been corrected programmatically.

---

### Is the dataset self-contained?
No. The metadata records link to external resources (e.g., Digital Commonwealth URLs pointing to media).

- **Are external resources stable?**  
  Not guaranteed. BPL hosts and maintains them but offers no permanence guarantees.
- **Are archival versions available?**  
  No. We recommend periodic scraping if long-term reproducibility is needed.
- **Are there usage restrictions?**  
  Yes. Each item’s license varies (public domain, Creative Commons, etc.). Usage should follow the rights noted in the `rights` field of each record when present.

---

### Does the dataset contain confidential or sensitive data?
No. All metadata is publicly available and part of open cultural collections.

---

### Could the dataset cause harm or discomfort?
Unlikely. However, some historical items may reference outdated or offensive language or subject matter due to the nature of historical archives.

---

### Is it possible to identify individuals?
Indirectly, yes. Some records may mention full names of historical figures or authors (e.g., in creator fields), but the data does not include personal identifiers or modern private information.

---

### Dataset Snapshot

| Property               | Value                                |
|------------------------|--------------------------------------|
| Size of dataset        | ~1.2 million metadata records        |
| Number of instances    | ~1.2 million                         |
| Number of fields       | Up to 135 fields (core ~8 used)      |
| Labeled classes        | N/A (not used for classification)    |
| Number of labels       | N/A                                  |

---

## ***Collection Process***

### What mechanisms were used to collect the data?
- **Method**: API-based scraping
- **Tool**: Python script (`load_script.py`)
- **Source**: [https://www.digitalcommonwealth.org/search.json](https://www.digitalcommonwealth.org/search.json)

---

### What was the sampling strategy?
For the live demo, we selected metadata from a **range of pages** in the API results. This ensured a diverse sample across collections, without manual filtering or stratification.

---

### Timeframe of data collection
- Metadata collection: Spring 2025
- The underlying metadata reflects items from the 1600s through 2000s
- The crawl was recent, but the data describes historical artifacts

---

## ***Preprocessing / Cleaning / Labeling***

### Was any preprocessing done?
Yes. Preprocessing steps included:
- Combining selected fields (`title`, `abstract`, `creator`, `date`) into a unified block for embedding
- Filtering out metadata with null/empty critical fields
- Normalizing text (removal of extra whitespace, special character fixes)

---

### Were transformations applied?
Yes:
- Removed newline/escape characters
- Filled missing fields with empty strings (if embedding required)
- Removed duplicate metadata entries by `id`

---

### Was raw data preserved?
Yes. Raw JSON metadata was stored in the `data/metadata_chunks/` folder.

---

### Is preprocessing code available?
Yes. Located in the GitHub repository under:
[https://github.com/BU-Spark/ml-bpl-rag/tree/dev/scripts](https://github.com/BU-Spark/ml-bpl-rag/tree/dev/scripts)

---

## ***Uses***

### What tasks has the dataset been used for?
- Retrieval-Augmented Generation (RAG)
- Document ranking and summarization
- Metadata-aware reranking using BM25
- Image and audio-based document discovery via metadata

---

### What other tasks could this dataset be used for?
- Named Entity Recognition (NER)
- Clustering or topic modeling of cultural collections
- Historical trend analysis (based on creator/date/subject metadata)

---

### Any risks for future use?
Yes. Inconsistent metadata formatting and missing values may impact downstream performance in tasks that require clean, structured data.

---

### Are there tasks for which this dataset should not be used?
- Tasks requiring raw media content (e.g., image classification, audio transcription)
- Privacy-sensitive applications (data is public, but not anonymized for modern individuals)

---

## ***Distribution***

### What access level should be given?
**External Open Access** – All metadata was collected from a public source and remains non-confidential.

---

## ***Maintenance***

### Can others contribute or extend the dataset?
Yes. The project is open-source, and additional metadata fields, modalities (e.g., OCR text), or collections can be added by:
- Extending the scraping logic
- Adding embedding pipelines for new modalities
- Updating Streamlit UI for new metadata fields

---

## ***Other***

No additional notes at this time.
