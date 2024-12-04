import streamlit as st
import os
import json
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Get the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

def load_json_file(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def setup_vector_store_from_json(json_data):
    """Create a vector store from JSON data."""
    documents = [Document(page_content=item["content"], metadata={"url": item["url"]}) for item in json_data]
    
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def setup_qa_chain(vector_store):
    """Set up the QA chain with a retriever."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

def main():
    # Set page title and header
    st.set_page_config(page_title="Football Players RAG App", page_icon="⚽")
    st.title("Football Players Knowledge Base 🏆")

    # Sidebar for initialization
    st.sidebar.header("Initialize Knowledge Base")
    if st.sidebar.button("Load Data"):
        try:
            # Load and preprocess the JSON file
            json_data = load_json_file("football_players.json")
            st.session_state.vector_store = setup_vector_store_from_json(json_data)
            st.session_state.qa_chain = setup_qa_chain(st.session_state.vector_store)
            st.sidebar.success("Knowledge base loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")

    # Query input and processing
    st.header("Ask a Question")
    query = st.text_input("Enter your question about football players:")

    if query:
        # Check if vector store and QA chain are initialized
        if st.session_state.qa_chain is None:
            st.warning("Please load the knowledge base first using the sidebar.")
        else:
            # Run the query
            try:
                response = st.session_state.qa_chain({"query": query})
                
                # Display answer
                st.subheader("Answer")
                st.write(response["result"])

                # Display sources
                st.subheader("Sources")
                sources = response["source_documents"]
                for i, doc in enumerate(sources, 1):
                    with st.expander(f"Source {i}"):
                        st.write(f"**Content:** {doc.page_content}")
                        st.write(f"**URL:** {doc.metadata.get('url', 'No URL available')}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
