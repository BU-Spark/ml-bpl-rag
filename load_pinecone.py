from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
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
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = sys.argv[4]
index = pc.Index(INDEX_NAME)

print("Loading JSON...")
meta = json.load(open(PATH))


model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

print("Initializing Pinecone index...")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)



text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,

    chunk_overlap=100,

    length_function=len,

    separators=["\n\n", "\n", " ", ""]

)

fields = ['abstract_tsi','title_info_primary_tsi','title_info_primary_subtitle_tsi', 'title_info_alternative_tsim']


print("Beginning Embeddings...")

start = time.time()

full_data = []

for page in meta:
    content = page['data']
    full_data += content
if BEGIN > END:
    slice = content[BEGIN:]
else:
    slice = content[BEGIN:END]

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
    print(documents)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    num += 1

end = time.time()
print(f"Embedded all documents in {end-start} seconds...")
