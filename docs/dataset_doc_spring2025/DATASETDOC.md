***Project Information*** 

* What is the project name?  
LibRAG - Boston Public Library
* What is the link to your project’s GitHub repository?   
https://github.com/BU-Spark/ml-bpl-rag/tree/dev
* What is the link to your project’s Google Drive folder? \*\**This should be a Spark\! Owned Google Drive folder \- please contact your PM if you do not have access\*\**  
https://drive.google.com/drive/folders/1X0ngdjpJyWnabFjFd0gONESmX_Zomqg1?usp=drive_link
* In your own words, what is this project about? What is the goal of this project?
This project, LIBRAG, is about building a smarter way to search the Boston Public Library’s massive digital archive using AI. Instead of relying on keyword search, which can be limiting or confusing, the goal is to let users ask natural language questions and get back relevant documents, images, or audio from the collection — along with helpful summaries and direct source links. It's designed to make historical information more accessible, accurate, and user-friendly for researchers, students, and the public.
* Who is the client for the project?  
Boston Public Library
* Who are the client contacts for the project?  
Eben English
* What class was this project part of?
Spark! Machine Learning X-Lab Practicum (CS549)

***Dataset Information***

* What data sets did you use in your project? Please provide a link to the data sets, this could be a link to a folder in your GitHub Repo, Spark\! owned Google Drive Folder for this project, or a path on the SCC, etc.  
We used metadata records from the Digital Commonwealth API, which provides access to digitized materials from libraries, museums, and historical societies across Massachusetts. The data includes structured metadata for over 1.2 million items, including images, audio recordings, text files, and more.

API Endpoint (for reference):
https://www.digitalcommonwealth.org/search.json

* Please provide a link to any data dictionaries for the datasets in this project. If one does not exist, please create a data dictionary for the datasets used in this project. **(Example of data dictionary)**   
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

This is a selected subset of metadata fields used. The full dataset includes up to 135 fields.

* What keywords or tags would you attach to the data set?  
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

  * Domain(s) of Application: Computer Vision, Object Detection, OCR, Image Classification, Image Segmentation, Facial Recognition, NLP, Topic Modeling, Sentiment Analysis, Named Entity Recognition, Text Classification, Summarization, Anomaly Detection, Other   
  * Sustainability, Health, Civic Tech, Voting, Housing, Policing, Budget, Education, Transportation, etc. 

  - **Natural Language Processing (NLP)**  
  - Semantic Search  
  - Summarization  
  - Named Entity Recognition (future extension)

- **Computer Vision (Metadata-level only)**  
  - Image Retrieval via metadata  
  - Image Preview Display  

- **Civic Tech**  
  - Public Archive Access  
  - Digital Humanities  
  - Library Tech Innovation  

*The following questions pertain to the datasets you used in your project.*   
*Motivation* 

* For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description. 

The metadata was created to catalog and preserve access to Massachusetts’s cultural and historical records. However, the default keyword search interface provided by the Digital Commonwealth lacks semantic understanding and contextual relevance.

**Our project’s goal** was to make this data more accessible through natural language queries by building a Retrieval-Augmented Generation (RAG) system that semantically retrieves and ranks the most relevant materials, while supporting modern features like image/audio display and source traceability.

*Composition*

* What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? What is the format of the instances (e.g., image data, text data, tabular data, audio data, video data, time series, graph data, geospatial data, multimodal (please specify), etc.)? Please provide a description.   
* How many instances are there in total (of each type, if appropriate)?  
* Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).  
* What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.   
* Is there any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include redacted text.   
* Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them  
* Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.   
* Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources,   
  * Are there guarantees that they will exist, and remain constant, over time;  
  * Are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created)?  
  * Are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a dataset consumer? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points as appropriate.   
* Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)? If so, please provide a description.   
* Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.   
* Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.   
* Dataset Snapshot, if there are multiple datasets please include multiple tables for each dataset. 


| Size of dataset |  |
| :---- | :---- |
| Number of instances |  |
| Number of fields  |  |
| Labeled classes |  |
| Number of labels  |  |


  
*Collection Process*

* What mechanisms or procedures were used to collect the data (e.g., API, artificially generated, crowdsourced \- paid, crowdsourced \- volunteer, scraped or crawled, survey, forms, or polls, taken from other existing datasets, provided by the client, etc)? How were these mechanisms or procedures validated?  
* If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?  
* Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created. 

*Preprocessing/cleaning/labeling* 

* Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section.   
* Were any transformations applied to the data (e.g., cleaning mismatched values, cleaning missing values, converting data types, data aggregation, dimensionality reduction, joining input sources, redaction or anonymization, etc.)? If so, please provide a description.   
* Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data, this could be a link to a folder in your GitHub Repo, Spark\! owned Google Drive Folder for this project, or a path on the SCC, etc.  
* Is the code that was used to preprocess/clean the data available? If so, please provide a link to it (e.g., EDA notebook/EDA script in the GitHub repository). 

*Uses* 

* What tasks has the dataset been used for so far? Please provide a description.   
* What (other) tasks could the dataset be used for?  
* Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?   
* Are there tasks for which the dataset should not be used? If so, please provide a description.

*Distribution*

* Based on discussions with the client, what access type should this dataset be given (eg., Internal (Restricted), External Open Access, Other)?

*Maintenance* 

* If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. 

*Other*

* Is there any other additional information that you would like to provide that has not already been covered in other sections?

