import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from langchain_openai import ChatOpenAI

# replace with file of your choosing
file = open("sample_full_text.json")
full_text = json.load(file)

# metadata csv file; should be included in repo
df_attributes = pd.read_csv("metadata_attributes.csv")

model = ChatOpenAI()

import re
def get_title(text):
    match = re.search(r'\d+\s+(.+?)\n', text)

    # Extracting and printing the title if there's a match
    if match:
        title = match.group(1)
    return title

# Turn the BPL data into a Document
from langchain.schema import Document

documents = []

for doc in full_text:
    # Extract metadata fields and apply get_title()
    title = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "title_info_primary_tsi"]))
    title_subtitle = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "title_info_primary_subtitle_tsi"]))
    title_alt = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "title_info_alternative_tsim"]))
    abstract = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "abstract_tsi"]))
    subject_facet = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "subject_facet_ssim"]))
    subject_geographic = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "subject_geographic_sim"]))
    genre = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "genre_basic_ssim"]))
    genre_specific = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "genre_specific_ssim"]))
    name_facet = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "name_facet_ssim"]))
    name_role = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "name_role_tsim"]))
    date_human = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "date_tsim"]))
    date_start = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "date_start_dtsi"]))
    date_end = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "date_end_dtsi"]))
    publisher = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "publisher_tsi"]))
    collection_name = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "collection_name_ssim"]))
    physical_location = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "physical_location_ssim"]))
    related_item_host = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "related_item_host_ssim"]))
    type_of_resource = get_title(str(df_attributes.loc[df_attributes["id"] == doc, "type_of_resource_ssim"]))
    URL = "https://www.digitalcommonwealth.org/search/" + get_title(str(df_attributes.loc[df_attributes["id"] == doc, "id"]))
    
    # Create Document with metadata
    documents.append(Document(
        page_content=full_text[doc]['text'],
        metadata={
            "title": title,
            "subtitle": title_subtitle,
            "title_alt": title_alt,
            "abstract": abstract,
            "subject_facet": subject_facet,
            "subject_geographic": subject_geographic,
            "genre": genre,
            "genre_specific": genre_specific,
            "name_facet": name_facet,
            "name_role": name_role,
            "date_human": date_human,
            "date_start": date_start,
            "date_end": date_end,
            "publisher": publisher,
            "collection_name": collection_name,
            "physical_location": physical_location,
            "related_item_host": related_item_host,
            "type_of_resource": type_of_resource,
            "URL": URL
        }
    ))

# Now for all of the vector store and reranking stuff
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# creating the vector store
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# now for the reranking step
weights = {
    "title": 1.0,
    "subtitle": 0.95,
    "title_alt": 0.9,
    "abstract": 0.85,
    "subject_facet": 0.8,
    "subject_geographic": 0.75,
    "genre": 0.7,
    "genre_specific": 0.65,
    "name_facet": 0.6,
    "name_role": 0.55,
    "date_human": 0.5,
    "date_start": 0.45,
    "date_end": 0.4,
    "publisher": 0.35,
    "collection_name": 0.3,
    "physical_location": 0.25,
    "related_item_host": 0.2,
    "type_of_resource": 0.15,
    "URL": 0.1
}

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

# our vector store:

# embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def compute_relevance_score(metadata_value, query):
    """
    Compute cosine similarity between the query and a metadata value using sentence-transformers.

    Args:
        metadata_value (str): The metadata value to compare.
        query (str): The query string.

    Returns:
        float: Cosine similarity score (between 0 and 1).
    """
    if not metadata_value or not query:
        return 0  # Return 0 if either the metadata or query is empty
    
    # Encode the metadata value and query into embeddings
    embeddings = model.encode([metadata_value, query], convert_to_tensor=False)  # Convert to NumPy
    metadata_embedding, query_embedding = embeddings

    # Compute cosine similarity
    similarity = cosine_similarity([metadata_embedding], [query_embedding])
    return similarity[0][0]  # Extract the scalar similarity value



def rerank_documents(documents, query, weights, vector_store, k=10):
    """
    Rerank documents based on metadata relevance scores and FAISS vector similarity scores.

    Args:
        documents (list): List of Document objects.
        query (str): The query string used for retrieval.
        weights (dict): Weights for each metadata field.
        vector_store (str): The vector store itself to get the similarity score

    Returns:
        list: Reranked documents in descending order of relevance.
    """

    final_score = 0

    reranked_results = []
    returned_docs = vector_store.similarity_search_with_score(query, k)
    for doc in returned_docs:
        final_score = doc[1]
        # Add weighted relevance scores for each metadata field
        for field, weight in weights.items():
            metadata_value = doc[0].metadata.get(field, "")  # Safely get metadata field value
            relevance_score = compute_relevance_score(metadata_value, query)
            final_score += weight * relevance_score

        reranked_results.append((doc, final_score))

    # Sort documents by the final score in descending order
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked_results]


docs = rerank_documents(documents, "Newspaper", weights, vector_store)

# now we should get an output like this for some k value:
# ('The Tocsin of Liberty', 'https://www.digitalcommonwealth.org/search/commonwealth:gf06jp23d', 'Reranked score: 1.1741459369659424')
docs_list = [(docs[i][0].metadata['title'], docs[i][0].metadata['URL'], f"Reranked score: {docs[i][1]}") for i in range(len(docs))]
docs_list.sort(key=lambda x: x[2], reverse=True)
for doc in docs_list:
    print(doc)
