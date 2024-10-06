# LibRAG Research Report

Members: Daniel Foley, Brandon Vargus, Jinanshi Mehta, Enrico Collautti

## Paper 1: [Retrieval Augmented Generation (RAG) and Beyond: A Comprehensive Survey on How to Make your LLMs use External Data More Wisely](https://arxiv.org/pdf/2409.14924)

- Proposes a method in which RAG queries are categorized and handled accordingly (explicit fact queries, implicit fact queries, interpretable rationale queries, and hidden rationale queries)  
- The first two categories are simpler than the rationale categories. They represent fact retrieval which is usually easier than rationale determination. (Intelligence vs. Wisdom from my understanding)  
- Also covers rationale behind RAG architecture that is probably useful  
  - Specifically addresses which techniques to explore given each query level (We can build our structure up sequentially to cover more complex queries)  
  - In addition to techniques for each query type, it references current research related to each technique.

## Paper 2: [AI-Powered Search: Embedding-Based Retrieval and Retrieval-Augmented Generation (RAG)] (https://dtunkelang.medium.com/ai-powered-search-embedding-based-retrieval-and-retrieval-augmented-generation-rag-cabeaba26a8b)

- Discusses the advantages of embedding words rather than using a bag of words, including retrieval and ranking techniques like HNSW graphs and machine-learned ranking.  
- Query Rewriting → Content Segmentation/Chunking → Retrieval   
- Extractive vs Abstractive summarization of documents

## Paper 3: [Contrastive Language Image Pretraining from OpenAI](https://openai.com/index/clip/)

- When you're dealing with multimodal inputs like videos, images, text, and audio, the embeddings need to be projected into a common space to ensure that the similarity computations are meaningful. The problem arises because different types of data have different structures (e.g., videos are spatiotemporal sequences, while text is sequential tokens), which can result in embeddings that are not directly comparable if treated independently. To resolve this, you need to ensure that all embeddings, regardless of modality, are projected into a shared latent space, where similarity can be meaningfully computed across modalities.  
- One solution is to use Pretrained Cross-Modal Models: models specifically designed for multimodal tasks, such as [CLIP](https://openai.com/index/clip/) (Contrastive Language-Image Pretraining). These models learn a shared embedding space for different types of data, ensuring that embeddings from one modality (e.g., an image) can be meaningfully compared to embeddings from another modality (e.g., text).

## Paper 4: [UNiversal Image-TExt Representation Learning](https://paperswithcode.com/method/uniter)

- Expanding on the problem described above, another solution is UNiversal Image-Text Representation, learned through large-scale pre-training over 4 image-text datasets, can help power multimodal embeddings.

## Open-Source: [LangChain](https://python.langchain.com/docs/introduction/)

- Open source framework to create applications using LLMs, which is useful for RAG.

## Notable Resource: SAIC Natural Language Library Querying

- Similar to our project in retrieval  
- Room for improvement in design (only returns resources rather than a structured response)