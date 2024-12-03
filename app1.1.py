from dotenv import load_dotenv  # Import dotenv to load environment variables
import os
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import json

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

# Global variables for vector store and QA chain
vector_store = None
qa_chain = None

# Step 1: Load and Process JSON Data
def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def setup_vector_store_from_json(json_data):
    # Create Document objects with URLs and content
    documents = [Document(page_content=item["content"], metadata={"url": item["url"]}) for item in json_data]

    # Create embeddings and store them in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def setup_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# Initialize Chainlit: Preload data when the chat starts
@cl.on_chat_start
async def chat_start():
    global vector_store, qa_chain

    # Load and preprocess the JSON file
    json_data = load_json_file("football_players.json")
    vector_store = setup_vector_store_from_json(json_data)
    qa_chain = setup_qa_chain(vector_store)

    # Send a welcome message
    await cl.Message(content="Welcome to the RAG app! Ask me any question based on the knowledge base.").send()

# Process user queries
@cl.on_message
async def main(message: cl.Message):
    global qa_chain

    # Ensure the QA chain is ready
    if qa_chain is None:
        await cl.Message(content="The app is still initializing. Please wait a moment and try again.").send()
        return

    # Get query from the user and run the QA chain
    query = message.content
    response = qa_chain({"query": query})

    # Extract the answer and source documents
    answer = response["result"]
    sources = response["source_documents"]

    # Format and send the response
    await cl.Message(content=f"**Answer:** {answer}").send()
    if sources:
        await cl.Message(content="**Sources:**").send()
        for i, doc in enumerate(sources, 1):
            url = doc.metadata.get("url", "No URL available")
            await cl.Message(content=f"**Source {i}:** {doc.page_content}\n**URL:** {url}").send()
