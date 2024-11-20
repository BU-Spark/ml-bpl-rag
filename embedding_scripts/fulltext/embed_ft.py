from langchain_huggingface import HuggingFaceEmbeddings

import faiss

from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_community.vectorstores import FAISS

from uuid import uuid4

from langchain_core.documents import Document

import json

from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import time
import sys

INDEX_PATH = "/projectnb/sparkgrp/ml-bpl-rag-data/vectorstore/faiss_index_ft"

BEGIN = int(sys.argv[1])
END = int(sys.argv[2])

index_file_path = os.path.join(INDEX_PATH,"index.faiss")

print("loading JSON...")

fulltext = json.load(open("/projectnb/sparkgrp/ml-bpl-rag-data/full_data/clean_ft.json"))



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


print("Beginning Embeddings...")
start = time.time()

if BEGIN < END:
    slice = list(fulltext)[BEGIN:END]
else:
    slice = list(fulltext)[BEGIN:]

for id in slice:

    print(id)

    documents = []

    chunks = text_splitter.split_text(fulltext[id])

    for chunk in chunks:

        doc = Document(

            page_content=chunk,

            metadata={"source": id},

        )

        documents += [doc]

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents,ids=uuids)
end = time.time()
print(f"Embedded all documents in {end-start} seconds...")

print(f"Saving vectorstore to {index_file_path}...")
vector_store.save_local(INDEX_PATH)

retriever = vector_store.as_retriever()
print("Performing test...\n\n")
print(retriever.invoke("Who wrote a letter to Z.B. Oakes?"))

