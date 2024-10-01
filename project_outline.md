# Technical Project Document Template

## Daniel Foley, Brandon Vargus, Jinanshi Mehta, Enrico Collautti
## 2024-September-29

## Overview

_In this document, based on the available project outline and summary of the project pitch, to the best of your abilities, you will come up with the technical plan or goals for implementing the project such that it best meets the stakeholder requirements._

### A. Provide a solution in terms of human actions to confirm if the task is within the scope of automation through AI.

*To assist in outlining the steps needed to achieve our final goal, outline the AI-less process that we are trying to automate with Machine Learning. Provide as much detail as possible.*

Read and remember everything in the library
Take in a request from a user
Recall what is relevant to the request from what you’ve read
Look up and find everything relevant and retrieve/deliver it to the user
Give a brief explanation of what you returned based on what you remember

### B. Problem Statement:

*In as direct terms as possible, provide the “Data Science” or "Machine Learning" problem statement version of the overview. Think of this as translating the above into a more technical definition to execute on. eg: a classification problem to segregate users into one of three groups on based on the historical user data available from a publicly available database*

1. (Optional) Increase robustness of data by performing OCR/speech recognition/multimodal model summarization
2. We will be using RAG techniques to accomplish this project
3. Run through entire BPL database, embed entire catalog
   a. Select vectorstore and ingestion method
   b. Run job that will ingest data into the vectorstore whilst chunking and embedding the full text data
      i. For multimedia, use a multimodal embeddings model in addition to the textual embedding
4. Ingest and process query and compare it to the database to look for similarities
   a. Cosine similarity search between vectors generated for chunks and the user’s query
      i. Potentially use an LLM to dissect query and generate alternative queries to compare against the vectorstore
5. Return entries we deemed similar and relevant
6. Generate brief summary of materials and provide links to relevant documentation
   a. Retrieve like vectors from the vectorstore and insert them into the prompt to the LLM that will generate a response


### C. Checklist for project completion

*Provide a bulleted list to the best of your current understanding, of the concrete technical goals and artifacts that, when complete, define the completion of the project. This checklist will likely evolve as your project progresses.*

1. Return relevant documents from BPL API
2. Return short description of materials and an answer to their query if applicable
3. Model that we can demo on HuggingFace.

### D. Outline a path to operationalization.

*Data Science Projects should have an operationalized end point in mind from the onset. Briefly describe how you see the tool produced by this project being used by the end user beyond a jupyter notebook or proof of concept. If possible, be specific and call out the relevant technologies that will be useful when making this available to the stakeholders as a final deliverable.*

## Resources

Data is hosted where it currently exists
Queries will run where they currently run

### Data Sets

Use API calls to get data; we could ask the client if we can get direct access to their database.
Potentially enrich certain data types

### References

1. https://gist.github.com/ebenenglish/7ebba7e662a37c64caae6a17080acafc
2.

## Weekly Meeting Updates


*Keep track of ongoing meetings in the Project Description document prepared by Spark staff for your project.*


Note: Once this markdown is finalized and merged, the contents of this should also be appended to the Project Description document.

