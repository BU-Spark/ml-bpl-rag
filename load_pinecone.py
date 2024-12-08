from langchain_huggingface import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore

from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter

import pinecone

import json

import os
from dotenv import load_dotenv
import sys

import time

load_dotenv()

BEGIN = int(sys.argv[1])

END = int(sys.argv[2])

PATH = sys.argv[3]

# Pinecone setup

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

INDEX_NAME = sys.argv[4]



print("Loading JSON...")

meta = json.load(open(PATH))



model_name = "sentence-transformers/all-MiniLM-L6-v2"

model_kwargs = {'device': 'cuda'}

encode_kwargs = {'normalize_embeddings': False}



print("Initializing Pinecone index...")

if INDEX_NAME not in pinecone.list_indexes():

    pinecone.create_index(

        name=INDEX_NAME,

        dimension=384,  # Dimension for all-MiniLM-L6-v2

        metric="cosine"

    )



vector_store = PineconeVectorStore(index=INDEX_NAME, embedding=embeddings)



text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,

    chunk_overlap=100,

    length_function=len,

    separators=["\n\n", "\n", " ", ""]

)



fields = ["table_of_contents_tsi", "title_info_primary_tsi", "title_info_primary_subtitle_tsi", "title_info_alternative_tsim", "abstract_tsi", "subject_facet_ssim", "subject_geographic_sim", "genre_basic_ssim", "genre_specific_ssim", "name_facet_ssim", "name_role_tsim", "date_tsim", "date_start_dtsi", "date_end_dtsi", "publisher_tsi", "collection_name_ssim", "physical_location_ssim", "related_item_host_ssim", "type_of_resource_ssim"]



print("Beginning Embeddings...")

start = time.time()



if BEGIN > END:

    slice = meta['data'][BEGIN:]

else:

    slice = meta['data'][BEGIN:END]



num = 0

for item in slice:

    id = item["id"]

    item_data = item["attributes"]

    print(id, time.time())

    documents = []

    for field in item_data:

        if (field in fields) or ("note" in field):

            entry = str(item_data[field])

            if len(entry) > 1000:

                chunks = text_splitter.split_text(entry)

                for chunk in chunks:

                    documents.append(Document(page_content=chunk, metadata={"source": id, "field": field}))

            else:

                documents.append(Document(page_content=entry, metadata={"source": id, "field": field}))



    if num % 1000 == 0:

        print(num, f"Added vectors to vectorstore at {time.time()} on id {id}")

    

    vector_store.add_documents(documents)

    num += 1



end = time.time()

print(f"Embedded all documents in {end-start} seconds...")



retriever = vector_store.as_retriever()

print("Performing test...\n\n")

print(retriever.invoke("Who wrote a letter to Z.B. Oakes?"))
