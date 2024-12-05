import streamlit as st
import os
import json
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Get the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
    
# takes a few minutes to load
file = open("bpl_data.json")

bpl = json.load(file)
    
#file = open("bpl_data.json")

#bpl = json.load(file)

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
    
    
    
def setup_qa_chain(vector_store):
    """Set up the QA chain with a retriever."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    return qa_chain

def setup_custom_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    docs = retriever.invoke()

def main():
    
    # Set page title and header
    st.set_page_config(page_title="LibRAG", page_icon="📚")
    st.title("Boston Public Library Database 📚")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Sidebar for initialization
    # st.sidebar.header("Initialize Knowledge Base")
    # if st.sidebar.button("Load Data"):
    #     try:
    #         st.session_state.vector_store = FAISS.load_local(
    #                     "vector-store", embeddings, allow_dangerous_deserialization=True
    #                     )
    #         st.session_state.qa_chain = setup_qa_chain(st.session_state.vector_store)
    #         st.sidebar.success("Knowledge base loaded successfully!")
    #     except Exception as e:
    #         st.sidebar.error(f"Error loading data: {e}")

    st.session_state.vector_store = FAISS.load_local("vector-store", embeddings, allow_dangerous_deserialization=True)
                        
    st.session_state.qa_chain = setup_qa_chain(st.session_state.vector_store)

    # Query input and processing
    st.header("Ask a Question")
    query = st.text_input("Enter your question about BPL's database")

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
                    source = doc.metadata["source"]
                    
                    abstract = None
                    
                    # find the specific source:
                    for j in range(len(bpl["Data"])):
                        ID = bpl['Data'][j]["id"]
                        if doc.metadata['source'] == ID:
                            abstract = bpl["Data"][j]['attributes']['abstract_tsi']
                            break
                        
                    with st.expander(f"Source {i}"):
                        st.write(f"**Content:** {abstract}")
                        st.write(f"**URL:** https://www.digitalcommonwealth.org/search/{doc.metadata['source']}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
