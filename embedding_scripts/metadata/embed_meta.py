from langchain_huggingface import HuggingFaceEmbeddings

import faiss

from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_community.vectorstores import FAISS

from uuid import uuid4

from langchain_core.documents import Document

import json

from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import sys
import time


BEGIN = int(sys.argv[1])
END = int(sys.argv[2])

INDEX_PATH = "/projectnb/sparkgrp/ml-bpl-rag-data/vectorstore/faiss_index_metadata"

index_file_path = os.path.join(INDEX_PATH,"index.faiss")

print("loading JSON...")

meta = json.load(open("/projectnb/sparkgrp/ml-bpl-rag-data/full_data/bpl_data.json"))

#model_name = "sentence-transformers/all-mpnet-base-v2"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}

encode_kwargs = {'normalize_embeddings': False}

hf = HuggingFaceEmbeddings(

    model_name=model_name,

    model_kwargs=model_kwargs,

    encode_kwargs=encode_kwargs

)

#read and load or instantiate a vectorstore
print("initializing index...")
if os.path.exists(index_file_path):
    print("\t case one")
    index = faiss.read_index(index_file_path)

    vector_store = FAISS.load_local(

        INDEX_PATH,hf,allow_dangerous_deserialization=True

    )

else:
    print("\t case two")

    index = faiss.IndexFlatL2(len(hf.embed_query("hello world")))

    vector_store = FAISS(

    embedding_function=hf,

    index=index,

    docstore=InMemoryDocstore(),

    index_to_docstore_id={},

    )


text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,

    chunk_overlap=100,

    length_function=len,

    separators=["\n\n","\n"," ",""]

)

fields = ["table_of_contents_tsi","title_info_primary_tsi","title_info_primary_subtitle_tsi","title_info_alternative_tsim","abstract_tsi","subject_facet_ssim","subject_geographic_sim","genre_basic_ssim","genre_specific_ssim","name_facet_ssim","name_role_tsim","date_tsim","date_start_dtsi","date_end_dtsi","publisher_tsi","collection_name_ssim","physical_location_ssim","related_item_host_ssim","type_of_resource_ssim"]

print("Beginning Embeddings...")
start = time.time()

if BEGIN > END:
    slice = meta['Data'][BEGIN:]
else:
    slice = meta['Data'][BEGIN:END]

for item in slice:
    id = item["id"]
    item_data = item["attributes"]
    print(id)
    documents = []
    for field in item_data:
        if (field in fields) or ("note" in field):
            entry = str(item_data[field])
            if len(entry) > 1000:
                chunks = text_splitter.split_text(entry)
                for chunk in chunks:
                    documents += [Document(page_content=chunk,metadata={"source":id,"field":field})]
            else:
                documents += [Document(page_content=entry,metadata={"source":id,"field":field})]

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents,ids=uuids)

end = time.time()
print(f"Embedded all documents in {end-start} seconds...")

print(f"Saving vectorstore to {index_file_path}...")
vector_store.save_local(INDEX_PATH)

retriever = vector_store.as_retriever()
print("Performing test...\n\n")
print(retriever.invoke("Who wrote a letter to Z.B. Oakes?"))

