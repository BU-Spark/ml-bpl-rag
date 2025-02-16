# BPL - Technical Project Document - Spring 2025

*Rithvik Nakirikanti, Jeremy Bui, Zuizz Saeed, Ananya Singh, 2025-02-16*

## Where We Started
We are building upon the work of a previous semester's implementation, which provided initial groundwork for metadata retrieval and embedding. This gives us a strong foundation to enhance and optimize the search experience further.

### A. Provide a solution in terms of human actions to confirm if the task is within the scope of automation through AI.

Before implementing an AI-driven solution, we need to define a manual workflow for automation. Since we are building on past work, we know automation is feasible. However, mapping out this workflow helps pinpoint areas for improvement, ensuring a more effective AI-driven retrieval and augmentation process.

To understand the automation scope, we will first document how a librarian or researcher would manually retrieve and contextualize search results.

#### **User Query Processing**
- A librarian reads and interprets the query.
- The query is refined based on historical search patterns and user intent.

#### **Data Retrieval**
- The librarian searches Digital Commonwealth using keyword-based metadata fields (title, author, subject, description).
- Full-text search is conducted if applicable.
- The librarian cross-references multiple sources, including books, newspapers, and image metadata.

#### **Validation and Relevance Ranking**
- The retrieved documents are manually reviewed for relevance.
- Filters such as publication date, location, and document type are applied.
- The librarian selects the most relevant results.

#### **Contextualization and Explanation**
- The librarian generates a short description of why each document is relevant.
- Additional background information is provided by referencing external sources if needed.
- The librarian structures a textual response for the user explaining the significance of the retrieved documents.

#### **Presentation of Results**
- The librarian presents search results in a structured format:
  - List of documents with titles, links, and metadata.
  - Summary of retrieved content.
  - Explanation of how each result relates to the original query.

#### **User Feedback and Refinement**
- If the user is unsatisfied, the librarian refines the query.
- New searches are performed based on user feedback.

## B. Problem Statement

This project focuses on improving the Boston Public Library’s digital repository search function. The goal is to make it easier for users to find relevant documents by allowing natural language queries and retrieving the most useful results. By enhancing the way search works, we aim to improve both accessibility and efficiency in navigating the library’s extensive digital collection.

To achieve this, we will explore ways to embed metadata alongside document text to improve search relevance. This will help ensure that important details about each document are factored into the results. Additionally, we will research how to assign different weights to sections of text, making sure that more meaningful content is prioritized in search rankings.

Another key part of this project is improving how documents are stored and retrieved using vector-based search techniques. We will evaluate various search APIs to find the most effective way to handle large-scale queries while keeping search speed fast. Additionally, we will look at ways to process data more efficiently and reduce runtime, ensuring the system remains responsive as more documents are added.

By combining these improvements, we aim to create a functional and efficient search tool that meets the needs of the Boston Public Library. This project will demonstrate how structured search enhancements can make digital archives easier to explore and use.

## C. Checklist for Project Completion

1. Return a list of results that answer the query that was inputted by the user, taking into account document metadata and weighing certain text over other text (demo on Hugging Face).
2. Improved performance of search engine from previous semester (better results, faster performance on same queries).
3. Search engine is compatible with image results (lower priority, as described by client).

## D. Outline a Path to Operationalization

### 1. Prototype Deployment
- Develop a Retrieval-Augmented Generation (RAG) model for document retrieval.  
- Host a demo on Hugging Face Spaces with Gradio for testing.  
- Research and experiment with different retrieval and ranking methods to ensure efficient search.  

### 2. Search Optimization
- Optimize search performance by evaluating metadata filtering and text weighting techniques.  
- Improve data processing efficiency to minimize latency.  

### 3. Usability and Integration
- Return search results with links and summaries for better accessibility.  
- Provide a simple, functional interface for user interaction.  
- Explore embedding the tool into the BPL website for future integration.  


## Final Deliverable
- A demo hosted on Hugging Face for testing.  
- Optimized retrieval and ranking for improved search results.  
- A scalable approach that can be further integrated into the BPL system.  


## Resources
- Application already exists on Hugging Face, and we have access to the Git repository for development and understanding.
- Digital Commonwealth API: [https://gist.github.com/ebenenglish/7ebba7e662a37c64caae6a17080acafc](https://gist.github.com/ebenenglish/7ebba7e662a37c64caae6a17080acafc)
- Theory of RAG: [https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2](https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2)

## References
- *What is RAG? - Retrieval-Augmented Generation AI Explained* – Amazon Web Services – <https://aws.amazon.com/what-is/retrieval-augmented-generation/>
- *Retrieval-Augmented Generation for Large Language Models: A Survey* – Yunfan Gao, Yun Xiong, Xinyu Gao, et al. – <https://arxiv.org/abs/2312.10997>
- *Using Hugging Face Integrations* – Gradio – <https://www.gradio.app/guides/using-hugging-face-integrations>
- *Retrieval-Augmented Generation for Natural Language Processing: A Survey* – Shangyu Wu, Ying Xiong, Yufei Cui, et al. – <https://arxiv.org/abs/2407.13193>
- *Building a Search Engine with Python and Elasticsearch* – PyCon 2018 – <https://www.youtube.com/watch?v=6_P_h2bDwYs>
- *Optimizing Vector Search with Metadata Filtering and Fuzzy Filtering* – KX Systems – <https://medium.com/kx-systems/optimizing-vector-search-with-metadata-filtering-41276e1a7370>

## Weekly Meeting Updates

### 02/14/2025 (Biweekly client meeting)
We talked with the client about focusing on metadata first, making sure we weight the important stuff for better search results. The plan is to match metadata with plain text from newspapers while keeping things efficient given the size of the data. We won’t spend much time on video or audio, but using images for embeddings and adding some image analysis could improve the results.

### 02/16/2025 (Weekly meeting with PM)
Tasks were assigned to all team members via Notion for easy tracking. A list of questions was prepared to clarify requirements with the stakeholder, and the project outline was discussed thoroughly with the Project Manager to identify key milestones.
