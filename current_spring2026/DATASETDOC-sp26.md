***Project Information*** 

* What is the project name?  
  **BPL-LIBRAG (BU Spark! BPL Hybrid RAG / GraphRAG project).**
* What is the link to your project’s GitHub repository?   
  **https://github.com/BU-Spark/ml-bpl-rag**
* What is the link to your project’s Google Drive folder? \*\**This should be a Spark\! Owned Google Drive folder \- please contact your PM if you do not have access\*\**  
  **https://drive.google.com/drive/folders/1DBfdDMxCzWbFtPjzjdS9S0yOgh0MpHOQ**
* In your own words, what is this project about? What is the goal of this project?   
  This project builds a hybrid retrieval-augmented generation (RAG) search system over Boston Public Library / Digital Commonwealth archival materials. The goal is to let users ask natural-language historical questions and receive grounded, cited answers based on retrieved archive records.
* Who is the client for the project?  
  **Boston Public Library (BPL), via the Digital Commonwealth ecosystem.**
* Who are the client contacts for the project?  
  **Eben English**
* What class was this project part of?
  **DS594**

***Dataset Information***

* What data sets did you use in your project? Please provide a link to the data sets, this could be a link to a folder in your GitHub Repo, Spark\! owned Google Drive Folder for this project, or a path on the SCC, etc.  
  1. **Full-text newspaper corpus (Boston Traveler)** in JSON format:  
     - <https://drive.google.com/drive/folders/18WGsu83qZhpZy_YCL9tuZgiPHvE7PRVN>
  2. **Metadata corpus from Digital Commonwealth API exports** in JSONL format:  
     - <https://drive.google.com/drive/folders/1-HUW-z42vgSstriLXm2PhpmE6RHcLyNR>
* Please provide a link to any data dictionaries for the datasets in this project. If one does not exist, please create a data dictionary for the datasets used in this project.   
  Data dictionary is derived from the ingestion mapping and DB schema:
  - Source-to-field mapping: `current_spring2026/ingestion/ingest.py`  
  - Canonical table schema: `current_spring2026/database/schema.py`  
  Core fields include identifiers (`ark_id`, `record_id`), bibliographic metadata (`title`, `issue_date`, `year`, `publisher`, `language`), provenance (`collection`, `institution`), access fields (`source_url`, `iiif_manifest`), and content fields (`clean_text`, `chunk_text`, embeddings).
* What keywords or tags would you attach to the data set?  
  * Domain(s) of Application: **NLP, Information Retrieval, Text Classification (query intent), Named Entity Recognition, Summarization, Graph-based Retrieval (GraphRAG), Digital Humanities**  
  * **Civic Tech, Education, Cultural Heritage, History, Archives, Libraries**

*The following questions pertain to the datasets you used in your project.*   
*Motivation* 

* For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description. 
  The dataset was assembled to support searchable, citation-grounded question answering over historical Massachusetts archival collections. A key gap addressed was combining heterogeneous records (full-text OCR newspapers and metadata-only archival records) into one retrieval pipeline that can still produce useful answers with provenance.

*Composition*

* What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? What is the format of the instances (e.g., image data, text data, tabular data, audio data, video data, time series, graph data, geospatial data, multimodal (please specify), etc.)? Please provide a description.   
  Instances primarily represent archival **documents/records** (newspaper issues/items/collections). Multiple representations are used: raw metadata JSON/JSONL, full-text strings, chunked text segments, and derived graph entities/edges for GraphRAG. Entity extraction is done with two methods: spaCy on metadata text, and GPT-4o-mini on the first 30,000 characters of full-text records.
* How many instances are there in total (of each type, if appropriate)?  
  From ingestion logs in this repo:
  - Full-text records ingested: **9,897**  
  - Full-text chunks created: **1,609,184**  
  - Metadata records ingested: **469,954**  
  (Note: metadata-only pass excludes record IDs already covered by full-text source.)
* Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).  
  It is a **partial working corpus**, not all possible Digital Commonwealth records. The full-text side is currently a selected `boston-traveler(1900-1946)` subset (years available in local JSON files), while the metadata has full coverage. This is representative enough for system development and evaluation, but not guaranteed statistically representative of the entire Digital Commonwealth archive.
* What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.   
  Both raw and derived data are present: raw metadata fields, clean OCR text, tokenized chunks, dense embeddings (BGE-M3 vectors), sparse lexical vectors (token IDs + weights), and query-time retrieval scores.
* Is there any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include redacted text.   
  Yes. Many records are metadata-only and therefore have no full text. Some OCR text is noisy or sparse. Some bibliographic fields are missing across source institutions, reflecting upstream archive variability.
* Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them  
  No fixed ML training/validation/test split is defined in this repository. For evaluation, query sets are maintained separately (e.g., `current_spring2026/test_queries.jsonl`) and run against the retrieval/generation pipeline.
* Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.   
  Yes: OCR errors and  historical spelling variation.
* Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources,   
  It relies on external archival resources (Digital Commonwealth/BPL item pages and IIIF manifests) linked in source metadata.  
  * Are there guarantees that they will exist, and remain constant, over time;  
    No hard guarantee is provided in this repo; links are expected to be stable but can change upstream.
  * Are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created)?  
    Local JSON/JSONL snapshots are preserved in project data folders, but external linked pages/manifests are maintained by source institutions.
  * Are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a dataset consumer? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points as appropriate.   
    Access and reuse terms may vary by Digital Commonwealth item/institution. Consumers should follow source-site rights statements for each record and associated media.
* Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)? If so, please provide a description.   
  Not to our knowledge; records are from public-facing archival collections.
* Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.   
  Potentially yes. Historical archives may include outdated/offensive language, sensitive historical topics, or disturbing events.
* Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.   
  Yes. Named individuals may appear in archival documents/metadata (e.g., historical figures, authors, publishers, persons referenced in news or records).
* Dataset Snapshot, if there are multiple datasets please include multiple tables for each dataset. 

**Dataset A: Full-text newspaper records (`data/fulltext/boston-traveler`)**

| Size of dataset | Value |
| :---- | :---- |
| Number of instances | 9,897 source records ingested into `documents` + 1,609,184 text chunks ingested into `chunks` |
| Number of fields  | `documents`: 28 columns total (27 feature/provenance fields + 1 PK `id`); `chunks`: 8 columns total (7 feature/linkage fields + 1 PK `id`) |
| Labeled classes | Not a labeled classification dataset |
| Number of labels  | N/A |

**Dataset B: Metadata records (`data/metadata/metadata.jsonl`)**

| Size of dataset | Value |
| :---- | :---- |
| Number of instances | 469,954 records |
| Number of fields  | 28 columns in canonical `documents` schema after mapping (including embeddings and tracking fields) |
| Labeled classes | Not a labeled classification dataset |
| Number of labels  | N/A |

  
*Collection Process*

* What mechanisms or procedures were used to collect the data (e.g., API, artificially generated, crowdsourced \- paid, crowdsourced \- volunteer, scraped or crawled, survey, forms, or polls, taken from other existing datasets, provided by the client, etc)? How were these mechanisms or procedures validated?  
  Data was collected directly from the Digital Commonwealth API in a staged pipeline: first paginated collection endpoints and date filters to gather record IDs, then fetched per-record metadata plus OCR text (`/search/{id}.json` and `/ark:/50959/{id}/text`) and wrote per-year JSON outputs, and harvested broad BPL metadata ( list-ids + per-record fetch) into JSONL. Validation was implemented via retry/backoff HTTP logic, resumable checkpoints, required-ID checks, deduplication of IDs, transcription/text-length gating, and progress/checkpoint logs with failed-ID tracking.
* If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?  
  Deterministic subset selection by available source files (e.g., specific `boston-traveler(1900-1946)` years and available metadata export files), not probabilistic random sampling.
* Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created. 
  Ingestion/preparation occurred during Spring 2026 development. The underlying archival content spans historical periods (for full-text local files, mainly early 1900s plus selected later years such as 1940 and 1946), so collection/processing time does not match original document creation time.

*Preprocessing/cleaning/labeling* 

* Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section.   
  Yes. Metadata records are normalized and mapped to a unified schema; HTML is stripped from abstract fields; noisy OCR text is cleaned with regex-based rules (e.g., dehyphenation, whitespace normalization, and non-text artifact removal); full-text is chunked into overlapping token windows; dense and sparse embeddings are generated; and metadata-only/full-text records are harmonized in shared database tables. For graph construction, entities are extracted with spaCy on metadata and with GPT-4o-mini on the first 30,000 characters of full-text content.
* Were any transformations applied to the data (e.g., cleaning mismatched values, cleaning missing values, converting data types, data aggregation, dimensionality reduction, joining input sources, redaction or anonymization, etc.)? If so, please provide a description.   
  Yes. Transformations include type normalization (lists, dates), HTML unescape/stripping for metadata text, regex-based OCR cleanup (dehyphenation, whitespace cleanup, and noisy character/artifact filtering), full-text chunking, and conversion to dense/sparse vector representations. Entity-extraction transformations were also applied (spaCy for metadata, GPT-4o-mini over first 30,000 full-text characters). Full-text and metadata-only sources are then harmonized into the shared schema using record identifiers.
* Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data, this could be a link to a folder in your GitHub Repo, Spark\! owned Google Drive Folder for this project, or a path on the SCC, etc.  
  Raw snapshots are retained for metadata in `current_spring2026/data/metadata/`. For full-text, OCR content is cleaned on the fly during processing and written as processed outputs (rather than preserved as a separate raw full-text archive). Parsed/processed representations are then written to PostgreSQL (`documents`, `chunks`) and Neo4j (graph entities/edges).
* Is the code that was used to preprocess/clean the data available? If so, please provide a link to it (e.g., EDA notebook/EDA script in the GitHub repository). 
  Yes. Main preprocessing/ingestion code is in `current_spring2026/ingestion/ingest.py` and `current_spring2026/ingestion/chunker.py`. Supporting schema and embedding code is in `current_spring2026/database/schema.py` and `current_spring2026/embedding/embedder.py`.

*Uses* 

* What tasks has the dataset been used for so far? Please provide a description.   
  It has been used for hybrid retrieval, query intent handling, GraphRAG expansion, citation-grounded answer generation, and offline/interactive evaluation of archival QA performance.
* What (other) tasks could the dataset be used for?  
  Historical trend analysis, temporal/geographic topic mining, entity-network analysis, OCR quality studies, and digital humanities research workflows.
* Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?   
  Yes. OCR noise, missing full text for many records, uneven temporal/collection coverage, and source-specific metadata quality can affect downstream model quality and retrieval fairness.
* Are there tasks for which the dataset should not be used? If so, please provide a description.
  It should not be used as a comprehensive or fully representative census of all historical records, nor for high-stakes factual decisions without independent verification against primary sources.

*Distribution*

* Based on discussions with the client, what access type should this dataset be given (eg., Internal (Restricted), External Open Access, Other)?
  **External Open Access**

*Maintenance* 

* If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. 
  Yes. Contributors can add new source dumps to `data/`, rerun ingestion (`python -m ingestion.ingest`), and rebuild graph artifacts as needed. The schema and pipeline are version-controlled, so updates can be tracked via standard Git workflows.

*Other*

* Is there any other additional information that you would like to provide that has not already been covered in other sections?
  This dataset and pipeline are intended for iterative research/development handoff across BU Spark! teams; logs, query history, and configurable retrieval parameters are included to support reproducibility and future tuning.
