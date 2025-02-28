***Project Information*** 

* The project name is LibRAG (Retrieval Augmented Generation)
* https://github.com/BU-Spark/ml-bpl-rag/tree/main   
* [Google Drive](https://drive.google.com/drive/folders/12_tsVcUgwdfUdXalD67NOgUL3tGeI6ss?usp=sharing)
* This project involved implementing natural language querying into the Digial Commonwealth project.
* Client: Boston Public Library
* Contact: Eben English 
* Class: DS549

***Dataset Information***

* Our data is contained on the SCC at /projectnb/sparkgrp/ml-bpl-rag-data  
  * /vectorstore/final_embeddings/metadata_index - faiss index for the metadata
  * /vectorstore/final_embeddings/fulltext_index - faiss index for the OCR text
  * /full_data/bpl_data.json - metadata
  * /full_data/clean_ft.json - fulltext
* We did not have formal datasets, instead we used the Digital Commonwealth API and created embeddings from it. There is no need for a data dictionary outside of [Digital Commonwealth API](https://github.com/boston-library/solr-core-conf/wiki/SolrDocument-field-reference:-public-API).
* What keywords or tags would you attach to the data set?  
  * Domain(s) of Application: Natural Language Processing, Library Science 
  * Civic tech

*The following questions pertain to the datasets you used in your project.*   
*Motivation* 

* We needed to create embeddings of the Digital Commonwealth's data in order to perform retrieval

*Composition*

* Each entry in the Digital Commonwealth API represents an object in their repo of varying format  
* There were ~1.3 million total objects last we checked, about 147,000 of which containing full-text from OCR'd documents. 
* Our data was a comprehensive snapshot, the API is being updated.
* Each field from the API represented metadata classifications   
* Data is publicly accessible and non-confidential
  
*Collection Process*

* We collected data from an API endpoint.
* No sampling was performed
* This data was collected in October 2024

*Preprocessing/cleaning/labeling* 

* Very limited character correction was performed on the fulltext data.
* No transformations were applied outside of embedding.
* The raw data is saved in ml-bpl-rag-data/full_data/bpl_data.json (metadata) clean_ft.json (fulltext)

*Uses* 

* Embedding for retrieval

*Distribution*

* This data is free to use and access by subsequent students of our project.

*Maintenance* 

There is currently no system in place for cleanly updating the data, though in our instructions within WRITEUP.md we include a way to ingest your own data from the API and embed it.

