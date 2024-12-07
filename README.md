# LIBRAG for Boston Public Library

## **Problem**

Our client, the Digital Repository Services Manager from the Boston Public Library (BPL), identified a need for a search function that enhances the accessibility and usability of the library's extensive digital resources. The goal is to implement a search function capable of understanding natural language queries and returning highly relevant documents from the BPL's vast database.

To meet this requirement, we are tasked with developing a demoable model that showcases this functionality. Our chosen approach leverages Retrieval-Augmented Generation (RAG) to provide an AI-driven, intuitive search experience.

---

## **Solution**

We are developing a Retrieval-Augmented Generation (RAG) model tailored to the Boston Public Library's needs. This solution will:

- **Enhance search performance**: Provide accurate and relevant responses to natural language queries.
- **Improve user experience**: Deliver intuitive interactions that bridge the gap between traditional library systems and modern AI solutions.
- **Showcase results through a user-friendly interface**: Display the LLM response alongside:
  - The documents used for generating the response.
  - Direct URLs to the source documents.

Our prototype uses a simplified RAG pipeline built with LangChain, designed to retrieve from a subset of the library's dataset.

---

## **How the Database Is Built**

The BPL database is vast and diverse, containing a wide variety of digital assets:

- **Total Items**: ~1.3 million items, including text, video, and audio.
- **Full-Text Documents**: ~130,000 documents (approximately 10% of the database).
- **Metadata**: JSON files totaling 6.7GB in size, containing 135 description fields.
- **File Types**:
  - **Still Images**: Most prevalent type, exceeding 600,000.
  - **Text Files**: Second most common type.
  - **Other File Types**: Fewer than 100,000 items, including notated music and manuscripts.
- **Text Abstracts**: Almost all images include textual descriptions, providing valuable metadata for retrieval.

The database is continuously updated, ensuring its relevance and comprehensiveness.

---

## **What is a RAG?**

Retrieval-Augmented Generation (RAG) combines information retrieval and generative AI to answer queries effectively. Here's how it works:

- **Query Processing**: Transform the user's natural language query into a format suitable for retrieval and embedding.
- **Document Retrieval**:
  - Break down large documents into smaller chunks (chunking) to enhance granularity.
  - Use embeddings to represent both the query and chunks as vectors.
  - Retrieve relevant texts based on similarity scores (e.g., cosine similarity).
- **Content Integration**:
  - Select and rank the most relevant chunks to answer the query.
- **Response Generation**: Use a large language model (LLM) to generate a response informed by the retrieved documents.

### Key Components of RAG:
- **Embedding Model**: Converts text into numerical vectors that capture semantic meaning.
- **Vector Store**: A database to store and retrieve embeddings for efficient similarity searches.

---

## **Where We Started**

Our initial prototype used:
- **LangChain** for building the RAG pipeline.
- **LangServe** for serving the UI.
- **Chroma** as the vector store.
- **OpenAI embeddings** for text representation.
- **GPT-4o** as the large language model.

---

## **Challenges**

Throughout development, we encountered several issues:
- **Limited Customization**: LangServe's UI was not sufficiently customizable.
- **Lightweight Vector Store**: Chroma couldn't handle our large dataset effectively.
- **High Costs**: OpenAI embeddings were too expensive for a database of this size.
- **Overpowered LLM**: GPT-4o was unnecessarily expensive for our use case.

---

## **Our Final Solution**

We refined our approach to address these challenges:
- **Embedding Model**: Switched to `all-MiniLM-L6-v2`, a free, open-source model from Hugging Face, offering excellent performance.
- **LLM**: Chose `4o-mini`, a cost-effective alternative with sufficient capabilities.
- **Vector Store**: Adopted FAISS for its efficiency and scalability with large datasets.
- **UI Development**: Replaced LangServe with Streamlit, which provided better customization and ease of use.

---

## **Deployment**

The final RAG model is hosted on Hugging Face, ensuring accessibility and reliability.

---

## **Ethical Assessment**

AI systems bring inherent risks that require careful management:
- **Biased Document Retrieval**:
  - Risk: The model may retrieve biased documents depending on the query.
  - Solution: Regularly audit retrieval processes to ensure unbiased results.
- **Conversational Hallucinations**:
  - Risk: The AI may generate responses not rooted in the database.
  - Solution: Restrict the system to generate responses strictly based on retrieved documents.

---

## **License**

This project is licensed under the **MIT License**, allowing others to freely integrate, modify, and build upon our work.

---
