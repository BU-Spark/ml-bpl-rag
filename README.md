---

title: LibRAG

emoji: 📖

colorFrom: purple

colorTo: gray

sdk: docker

app_port: 7860

---

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

The Digital Commonwealth database is vast and diverse, containing a wide variety of digital assets:

- **Total Items**: ~1.3 million items, including text, video, and audio.
- **Full-Text Documents**: A subset of the total items, ~147,000 OCR documents.
- **Metadata**: JSON files containing up to 135 description fields.
- **File Types**:
  - **Still Images**: Most prevalent type, exceeding 600,000.
  - **Text Files**: Second most common type.
  - **Other File Types**: Fewer than 100,000 items, including notated music and manuscripts.
- **Text Abstracts**: Almost all images include textual descriptions, providing valuable metadata for retrieval.

Our solution ingested metadata items through the Digital Commonwealth API and embedded them for querying.

---

## **What is RAG?**

Retrieval-Augmented Generation (RAG) combines information retrieval and generative AI to answer queries over a bespoke dataset. Here's how it works:

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
- **Vector Store**: A database to store and retrieve embeddings for similarity searches.
- **Rerankig Algorithm**: Determines priority of retrieved documents from the vectorstore (a second retrieval of sorts).

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
- **Lightweight Vector Store**: Chroma and FAISS couldn't handle our large dataset effectively.
- **Local Vector Store**: A local vectorstore required the local storage of our large data and were cumbersome to load.
- **High Costs**: OpenAI embeddings were too expensive for a database of this size. Pinecone expenses grow quickly so we used a subset of our data.

---

## **Our Final Solution**

We refined our approach to address these challenges:
- **Embedding Model**: Switched to `all-MiniLM-L6-v2`, a free, open-source model from Hugging Face, offering excellent performance.
- **LLM**: Chose `4o-mini`, a cost-effective alternative with sufficient capabilities.
- **Vector Store**: Adopted Pinecone vectorstore due to its accessibility via API given the size of the data.
- **UI Development**: Using Streamlit for UI due to its easy integration through python and compatibility with Huggingface Spaces.

---

## **Deployment**

The final RAG model is hosted on Hugging Face, ensuring accessibility and reliability.

Our current solution externally hosts data on Pinecone and calls GPT-4o-mini.

---

## **Ethical Assessment**

AI systems bring inherent risks that require careful management:
- **Biased Document Retrieval**:
  - Risk: The model may retrieve biased documents depending on the query.
  - Solution: Present to the user that this is from doocuments retrieved, not a response on behalf of the BPL.
- **Conversational Hallucinations**:
  - Risk: The AI may generate responses not rooted in the database.
  - Solution: Restrict the system to generate responses strictly based on retrieved documents.

---

## **License**

This project is licensed under the **MIT License**, allowing others to freely integrate, modify, and build upon our work.

---

## **Replication**

Let's get it running. For this use-case, we will only be ingesting the metadata associated with each record. Note that the general infrastructure would work on the OCR text as well.

**Begin by creating a python virtual environment**

Make sure you are at least in Python 3.12.4

```
python -m venv .venv
```
```
source .venv/bin/activate (Mac)
or
.venv/Scripts/Activate (Windows)
```
```
python -m pip install -r requirements.txt
```

Once you have your environment, it is time to begin running scripts!

**Loading the BPL Digital Commonwealth metadata**
BEGIN = page number to start on
END = page number to end on (non-inclusive)
Pages each contain 100 items from the digital repository

```
python load_script.py <BEGIN> <END>
```

This will load the specified chunk of items from the Digital Commonwealth API, outputting to a file called "out\<BEGIN\>_\<END\>.json" in your current directory.

Next, we need to initialize our vectorstore

**Initializing our Vectorstore**

There are many vectorstore options to choose from. We ultimately decided on Pinecone due to the size of the data, which made local vectorstores inhibitively slow.

For our approach:

Make a Pinecone.io account

Navigate to Databases > Indexes and make a serverless Index. This is where we will store our embeddings. Make the length of each vector equal to whatever embeddings model you use. We used all-MiniLM-L6-v2, so our vector length was 384.

You will also want to create and store a Pinecone API Key in your .env file.

```
PINECONE_API_KEY = <your-api-key-here>
```

**Embedding our data**
BEGIN = Start index of the metadata to begin embedding
END = Final index of the metadata to stop embedding (non-inclusive)
PATH = Path to your .json file of metadata
INDEX = Name of your pinecone index

```
python load_pinecone.py <BEGIN> <END> <PATH> <INDEX>
```

Check to make sure the embeddings are being loaded into your index.

**Running the App**
Now that you have your embeddings, you're ready for the main event!

Our implementation uses GPT-4o-mini from OpenAI, however you could fit in your own model to the RAG.py file. With our implementation, you will need to enter an openai api key into the .env file.

Also, make sure to replace the variable INDEX_NAME in streamlit-app.py with the name of your index.

```
OPENAI_API_KEY = "<your-api-key>"
```

Once you do that, you're ready to run.

```
streamlit run streamlit_app.py
```

This will run the app on port 8501. When querying, please be patient, sometimes the retrieval and re-ranking process is slow depending on how much data you embedded.\

**On-going Challenges**
 - Vector Store: 1.3 million metadata objects and 147,000 full text docs resulted in a cumulative ~140GB of vectors when using all-MiniLM-L6-v2 and recursive character splitting on 1000 characters. This made locally hosted vectorstores cumbersome and implausible. Our solution was to migrate a portion of our metadata vectors to Pinecone and used that in our final implementation. Hosting on Pinecone can become expensive and adds another dimension of complexity to the project.
 - Speed: Currently, the app takes anywhere from 25-70sec to generate a response, we have found that the most time-consuming aspect of this is our calls to the Digital Commonwealth API to retrieve the rest of the metadata for each object retrieved within Pinecone. We were unable to associate an object's full metadata in Pinecone due to internal limits, so we are hitting the Digital Commonwealth API to do so. On average, responses take 1/4 of a sec, however across 100 responses that becomes cumbersome.
 - Query Alignment: The way queries are worded can have impact on the quality of retrieval. We attempted to implement a form of query alignment by using an llm to generate a sample response to the query, however we found it to be ineffective and detrimental. One specific aspect is the efficacy of specific queries versus vague ones ("Who wrote letters to WEB Du Bois?" vs "What happened in Boston in 1919?"). Queries as a whole may benefit from segmentation into the likely metadata fields they contain in order to inform querying (set up separate vectorstores for each field and then retrieve different parts of the query respectively). Further research should be done in this area to improve standardization of query alignment ot improve retrieval.
 - Usage Monitoring: Real-time usage monitoring through the console logs is implemented, however it would be beneficial to implement a form of persistent usage monitoring for generating insights into model performance and query wording for the purpose of ML/OPs.

 **Ad Hoc Process/Recommendations**
  - Our Demo on huggingface (that will temporarily be hooked up to a group member's personal Pinecone index before being disconnected after submission) only included retrieval over 600k entries in the Digital Commonwealth API. Each entry's title fields and abstract were embedded and input into the vectorstore. We first retrieve the top 100 related vectors to the query (with the intent to reduce vectorstore size and only retrieve on topical relevance), then we retrieve the metadata for certain fields from each retrieved vector's source id deemed related to queries (abstract, title, format, etc.) and rerank with BM25 off of that (with the intent to then prioritize entries on metadata like format and date). This was a way to effectively put together a quick demo.
  - The size of the data is significant in size and largely grows with vectorsize assuming you are significantly chunking each entry. It is our formal recommendation that you host your vectorstore on Pinecone or another service for efficient retrieval and initialization as well as in consideration of the storage of huggingface spaces.
  - As mentioned previously, a way to segment and analyze each query prior to retrieval could create a more reliable and accuracte retriever. Also of not is our prompt engineering. We strongly suggest using XML tags and a parser for efficient Chain of Thought in order to minimize llm calls.
  - Currently we are linearly hitting the Digital Commonwealth API for metadata once we retrieve the top 100 vectors in order to perform reranking and contextual addition to the prompt. This is really slow. We recommend that you either forego this method for some other or parallelize your calls (we tried parallelization, however found that rate limiting was too severe). A solution might be to create a metadata database and initialize it only on startup for referencing or to create proxies for api parallelization.

  Thank You and Best of Luck!

Continuing Development: Audio and Image Query Results
Recent updates have added major improvements to handling audio and image results from the Digital Commonwealth metadata.

Summary of Current Features
Audio Metadata Grouping:

A custom script (group_audio_metadata.py) retrieves and groups separate metadata fields (e.g., title, description, notes) that belong to the same source_id.

This allows full reconstruction of individual audio records despite Pinecone storing fields separately.

Grouped audio results are now available to be retrieved and presented through the Streamlit app.

Image Retrieval and Display:

Images associated with records are retrieved from the Digital Commonwealth URLs.

Downloaded images are displayed inside the chat UI alongside the document's title and description.

UI Enhancements:

The Streamlit app was updated to recognize audio entries and display them with an appropriate 🔊 icon.

Future support for audio players (e.g., st.audio()) is ready to be plugged in once direct audio URLs are available.

Image displays were enhanced to better match the look and feel of BPL's original digital repository.

Future Engineering Directions
To continue or improve the project from here, the following areas are recommended:

1. Full Audio Player Integration
Current state: Audio entries are labeled but do not embed playable audio yet.

Next step: Identify and extract actual audio file URLs (if available) from Digital Commonwealth or metadata fields.

Implementation idea: Use st.audio(url) to directly embed playable audio clips into the Streamlit app.

2. Smarter Multimedia Querying
Current state: Queries retrieve relevant entries, but audio/image prioritization is basic.

Next step: Add format-aware reranking — e.g., prioritize or separate "Audio" results when queries mention "listen", "hear", "recordings", etc.

Implementation idea: Build a simple query classifier to detect if a query is asking for audio or visual content, and then boost those entries accordingly.

3. Metadata Preloading and Caching
Current state: Each API call retrieves metadata individually (slow).

Next step:

Build a pre-cached metadata database (e.g., a local SQLite DB or a hosted lightweight database) initialized once at startup.

Future retrievals could hit this database instead of calling the API repeatedly, massively speeding up the app.

4. Parallel API Calls for Metadata
Current state: Metadata enrichment is sequential, causing slowdowns.

Next step: Implement parallel requests with rate limiting awareness or a batch API retrieval method (if supported).

Implementation idea: Use asyncio + backoff retries to batch metadata pulls without exceeding API limits.

5. Expand Grouping Logic to Other Formats
Current state: Grouping is focused mainly on audio records.

Next step: Extend grouping scripts to other types (e.g., manuscripts, video) by dynamically adjusting which fields to combine for each media type.
