***Project Information***

* What is the project name?  
  * LIBRAG: Retrieval-Augmented Search for Boston Public Library

* What is the link to your project’s GitHub repository?   
  * https://github.com/BU-Spark/ml-bpl-rag

* What is the link to your project’s Google Drive folder? **This should be a Spark! Owned Google Drive folder - please contact your PM if you do not have access**  
  * [Spark-owned Google Drive folder](https://drive.google.com/drive/folders/1bR8FCIilwfyKEUWfxwua5h0cDsgjxPOl?dmr=1&ec=wgc-drive-hero-goto) – please contact the Spark! PM for this course section to obtain or update the official link.

* In your own words, what is this project about? What is the goal of this project?   
  * This project builds a retrieval-augmented generation (RAG) system to make the Boston Public Library’s Digital Commonwealth collections easier to explore. Instead of relying on brittle keyword search, our system lets users ask natural language questions (e.g., “photos of Boston in the 1920s” or “maps of Worcester from the 18th century”) and returns relevant catalog entries from BPL’s digital collections along with a librarian-style summary of what materials are available. The goal is to support historians, educators, students, and the general public in discovering relevant archival materials more efficiently and intuitively.

* Who is the client for the project?  
  * Boston Public Library (BPL) – Digital Services / Digital Collections

* Who are the client contacts for the project?  
  * Project contacts are the BPL Digital Services team and Spark! staff (see course documentation / PM notes for specific names and emails).

* What class was this project part of?  
  * DS/CS 549 – Machine Learning (Boston University)


***Dataset Information***

* What data sets did you use in your project? Please provide a link to the data sets, this could be a link to a folder in your GitHub Repo, Spark! owned Google Drive Folder for this project, or a path on the SCC, etc.  
  * Primary dataset: **Digital Commonwealth metadata for items physically held at Boston Public Library**. These records are pulled via the Digital Commonwealth search API and stored in a PostgreSQL database with a Bronze–Silver–Gold schema:
    * `bronze.bpl_metadata` – raw JSONB metadata from the API
    * `silver.bpl_combined` – processed, concatenated metadata fields with parsed date ranges
    * `gold.bpl_embeddings` – chunked text + vector embeddings (pgvector)
  * Raw JSON files (used to populate Bronze) are written to `current_fall2025/data/raw/` (on SCC / local clone, not committed to GitHub).
  * The scripts that operate on these datasets live in `current_fall2025/scripts/` (see `load_scraper_bpl.py`, `bronze_load_bpl_data_to_pg.py`, `silver_build_bpl_combined_table.py`, and `gold_build_bpl_vector.py`).

* Please provide a link to any data dictionaries for the datasets in this project. If one does not exist, please create a data dictionary for the datasets used in this project. **(Example of data dictionary)**   
  * A brief schema/data dictionary for the three main tables is provided in `current_fall2025/dataset-documentation/DATA_DICTIONARY-fa25.md` (fields, types, and descriptions for `bronze.bpl_metadata`, `silver.bpl_combined`, and `gold.bpl_embeddings`).

* What keywords or tags would you attach to the data set?  
  * Domain(s) of Application:  
    * NLP  
    * Information Retrieval / RAG  
    * Summarization (catalog-style summaries)  
  * Application Areas:  
    * Civic Tech  
    * Libraries / Archives  
    * Education  
    * Digital Humanities / History


*The following questions pertain to the datasets you used in your project.*  
*Motivation* 

* For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.   
  * Digital Commonwealth and BPL originally created this metadata to describe and provide access to digitized collection items (photographs, manuscripts, maps, audio, etc.) for public discovery. For our project, we aggregated and shaped this metadata into a form suitable for semantic search and retrieval-augmented generation. The specific gap we are addressing is that the existing keyword-based search interface makes it hard for non-expert users to find relevant materials, especially when they do not know exact titles or controlled vocabulary terms.


*Composition*

* What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? What is the format of the instances (e.g., image data, text data, tabular data, audio data, video data, time series, graph data, geospatial data, multimodal (please specify), etc.)? Please provide a description.   
  * Each instance in the core metadata dataset represents a **single Digital Commonwealth item** that is physically held at the Boston Public Library (e.g., a photograph, map, manuscript, audio recording, or other digitized object).  
  * The primary format we work with is **tabular/JSON metadata** (text fields and arrays of strings) stored as JSONB in PostgreSQL and as rows in derived tables.  
  * The records contain descriptive fields (titles, abstracts, subjects, people, geographic locations, dates, collection names, resource types, etc.) and links to the underlying digital objects (images, audio, etc.) hosted on Digital Commonwealth.

* How many instances are there in total (of each type, if appropriate)?  
  * Approximately **1.2 million** metadata records for BPL-held items were ingested from Digital Commonwealth, which then expand to several million text chunks in the embeddings table after splitting for vectorization.

* Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).  
  * The dataset is intended to include **all Digital Commonwealth records where Boston Public Library is the physical holding institution**, as returned by the public API at the time of collection (subject to API filters and any upstream changes). The larger set would be the full Digital Commonwealth corpus across all contributing institutions. Our subset is not meant to be a random sample; it is a **coverage-based slice** focused on BPL’s holdings. Within that slice, we do not intentionally omit records, but incompleteness is possible due to API availability, occasional failures, or future additions by BPL/DC.

* What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.   
  * **Bronze layer (`bronze.bpl_metadata`)**: raw metadata JSON returned by the Digital Commonwealth API, including nested attributes.  
  * **Silver layer (`silver.bpl_combined`)**: a concatenated text field (`summary_text`) built from key metadata fields (title, subtitle, abstract, notes, subjects, people, geographic subjects, dates, resource types, collection names), plus parsed `date_start`/`date_end` ranges.  
  * **Gold layer (`gold.bpl_embeddings`)**: short text chunks derived from `summary_text`, along with 768-dimensional vector embeddings and a copy of the metadata JSONB. These are feature representations for vector search rather than new raw data.

* Is there any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include redacted text.   
  * Many metadata fields are sparsely populated or missing for some records (e.g., no abstract, incomplete subject headings, missing date fields, or limited geographic information). This is due to how the original catalog records were created and curated over time, not because we removed information. Our pipeline treats missing fields as empty strings but otherwise preserves the structure of the original metadata.

* Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.  
  * We do **not** define standard train/validation/test splits on the metadata itself. Instead, evaluation is done at the query level using separate test query files (e.g., `current_fall2025/evaluation/test_queries.json`) while treating the underlying corpus as a single retrieval database.

* Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.   
  * Common issues include:  
    * Inconsistent or approximate date strings (e.g., "ca. 1910"), which we normalize into ranges.  
    * Occasional duplicates or near-duplicates where multiple records represent related items or different digital manifestations.  
    * Variation in subject headings and naming conventions across time and cataloging practices.  
  * These are inherent to the source catalog and not introduced by our processing.

* Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources,
  * The metadata is largely self-contained for retrieval purposes, but **many fields include identifiers and URLs that reference Digital Commonwealth records and media** (e.g., image viewers, audio players). The long-term availability of those external resources is controlled by BPL and Digital Commonwealth.
  * Are there guarantees that they will exist, and remain constant, over time;  
    * There are no hard guarantees, but Digital Commonwealth and BPL treat these as production services intended to be maintained over time. URLs and API behavior could change in the future as systems evolve.  
  * Are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created)?  
    * We do not create our own official archival snapshot of all binary media; we store and version the **metadata** in PostgreSQL and raw JSON files. The underlying images/audio remain hosted by Digital Commonwealth/BPL.  
  * Are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a dataset consumer? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points as appropriate.   
    * Use of the metadata and linked digital objects is subject to BPL and Digital Commonwealth terms of use and any rights statements attached to individual items. Users of this dataset should consult those terms before reusing images or audio beyond research/educational purposes.

* Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)? If so, please provide a description.   
  * No. The dataset is built entirely from **publicly accessible Digital Commonwealth catalog records** and does not include private communications or restricted-access data.

* Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.   
  * Potentially yes. Because this is a large historical collection, some items (and their metadata descriptions) may depict or reference sensitive topics, including war, racial violence, colonialism, or other forms of discrimination and harm. These reflect the underlying archival materials, not any labels added by our team.

* Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.   
  * The metadata can reference named individuals (e.g., creators, photographers, subjects, or people depicted in images). In many cases these are historical figures, but in some cases they could be identifiable individuals. This information originates from the catalog records and is not newly inferred by our system.

* Dataset Snapshot, if there are multiple datasets please include multiple tables for each dataset.  


| Dataset / Table           | Size (approx.) | Number of instances | Number of fields (logical) | Labeled classes | Number of labels |
| :------------------------ | :------------- | :------------------ | :-------------------------- | :-------------- | :--------------- |
| `bronze.bpl_metadata`    | ~8–10 GB       | ~1.2M records       | 1 JSONB column (many keys) | N/A             | N/A              |
| `silver.bpl_combined`    | ~4–6 GB        | ~1.2M records       | 4–5 main columns           | N/A             | N/A              |
| `gold.bpl_embeddings`    | ~15+ GB        | ~3–5M chunks        | text + vector + metadata   | N/A             | N/A              |


*Collection Process*

* What mechanisms or procedures were used to collect the data (e.g., API, artificially generated, crowdsourced - paid, crowdsourced - volunteer, scraped or crawled, survey, forms, or polls, taken from other existing datasets, provided by the client, etc)? How were these mechanisms or procedures validated?  
  * We use the **public Digital Commonwealth search API** with filters for Boston Public Library as the physical location. A Python script (`current_fall2025/scripts/load_scraper_bpl.py`) paginates through the API, collects records, and writes them to JSON. We validated this mechanism by spot-checking records against the Digital Commonwealth web interface and confirming that the same items and metadata appear in both places.

* If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?  
  * The dataset is a **deterministic subset**: all records returned by the API when filtering to BPL as physical location. There is no random sampling; any omissions would be due to API limits or errors rather than an explicit sampling strategy.

* Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.   
  * The metadata was collected during the **Spring 2025 and Fall 2025** semesters for this course project. The underlying items themselves span several centuries (17th–23rd centuries), so the collection timeframe does not match the creation timeframe of the materials; rather, we are taking a contemporary snapshot of BPL’s digital catalog at that point in time.


*Preprocessing/cleaning/labeling* 

* Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section.   
  * Yes. We perform several preprocessing steps:  
    * Build a combined text field (`summary_text`) from multiple metadata fields for better semantic embedding.  
    * Extract 4-digit years from date strings and normalize approximate dates (e.g., "ca.") into ranges.  
    * Chunk the combined text into overlapping segments (~1000 characters) for embedding.  
    * Generate vector embeddings using the `sentence-transformers/all-mpnet-base-v2` model and store them in `gold.bpl_embeddings` via pgvector.

* Were any transformations applied to the data (e.g., cleaning mismatched values, cleaning missing values, converting data types, data aggregation, dimensionality reduction, joining input sources, redaction or anonymization, etc.)? If so, please provide a description.   
  * Transformations include:  
    * Converting nested JSON arrays into flattened text using helper SQL functions (e.g., `flatten_jsonb_array`).  
    * Generating numeric `date_start`/`date_end` fields from free-text dates for range filtering.  
    * Aggregating various descriptive fields into a single text string to feed into the embedding model.  
    * No explicit redaction or anonymization is applied beyond what is present in the original metadata.

* Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data, this could be a link to a folder in your GitHub Repo, Spark! owned Google Drive Folder for this project, or a path on the SCC, etc.  
  * Yes. Raw metadata is retained in two forms:  
    * JSON files in `current_fall2025/data/raw/` (on SCC/local environment).  
    * The `bronze.bpl_metadata` table in PostgreSQL, which stores the original JSONB records from the API.  
  * These can be reused to regenerate Silver/Gold layers if we change our processing pipeline.

* Is the code that was used to preprocess/clean the data available? If so, please provide a link to it (e.g., EDA notebook/EDA script in the GitHub repository).   
  * Yes, all preprocessing and loading code is in the repository under `current_fall2025/scripts/`:  
    * `load_scraper_bpl.py` – scrape metadata from Digital Commonwealth API  
    * `bronze_load_bpl_data_to_pg.py` – load raw JSON into PostgreSQL Bronze table  
    * `silver_build_bpl_combined_table.py` – build processed Silver table  
    * `gold_build_bpl_vector.py` – generate embeddings and populate Gold table  
    * SQL definitions are in `current_fall2025/db/`.


*Uses* 

* What tasks has the dataset been used for so far? Please provide a description.   
  * The dataset is used to power a **semantic search and RAG system** that:  
    * Retrieves relevant catalog entries for natural language queries using vector search + metadata filters.  
    * Reranks results using BM25 and metadata-based scoring.  
    * Generates high-level, librarian-style summaries of what relevant materials exist in the BPL collections.

* What (other) tasks could the dataset be used for?   
  * Potential additional uses include:  
    * Collection-level analytics (e.g., how coverage varies by time period, location, or subject).  
    * Training supervised models for subject/genre classification.  
    * Clustering items into thematic groups for exhibit curation.  
    * Supporting recommendation systems for related archival materials.

* Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?   
  * Because we rely on the public API and focus on BPL-held items, the dataset reflects **current cataloging practice and API availability** at the time of collection. Missing or inconsistent metadata fields, especially dates and subjects, can impact models that depend heavily on those signals. The embedding layer is also tied to the specific model used (`all-mpnet-base-v2`), so downstream systems should be aware of that when comparing vectors or refreshing embeddings.

* Are there tasks for which the dataset should not be used? If so, please provide a description.  
  * The dataset should **not** be used for high-stakes decision-making about individuals (e.g., employment, credit, legal judgments) or for drawing sensitive inferences about people based solely on historical metadata. It also should not be treated as a fully comprehensive or unbiased representation of all historical materials related to a topic; it only reflects items that have been digitized and cataloged by BPL and partners.


*Distribution*

* Based on discussions with the client, what access type should this dataset be given (eg., Internal (Restricted), External Open Access, Other)?  
  * The underlying metadata is derived from **publicly accessible Digital Commonwealth records**, so the dataset is effectively **External Open Access** for research and educational purposes, subject to BPL/Digital Commonwealth terms of use and rights statements for individual items.


*Maintenance* 

* If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description.  
  * Yes. New snapshots of the metadata can be generated by re-running the scraper and loading pipelines in `current_fall2025/scripts/`. Contributions to the code or schema can be made via pull requests to the GitHub repository. Any substantial changes to how we represent or process the metadata should be documented in this dataset documentation and the technical manifest.


*Other*

* Is there any other additional information that you would like to provide that has not already been covered in other sections?  
  * When interpreting results from models trained or evaluated on this dataset, it is important to remember that **archival collections are themselves shaped by institutional priorities and historical biases**. The absence of materials in a given area does not imply that events or communities were unimportant; it may simply reflect what has (or has not) been digitized and cataloged to date.
