import streamlit as st
import os
from typing import List, Tuple, Optional
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from RAG import RAG
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Boston Public Library Chatbot",
    page_icon="🤖",
    layout="wide"
)

def initialize_models() -> Tuple[Optional[ChatOpenAI], HuggingFaceEmbeddings]:
    """Initialize the language model and embeddings."""
    try:
        load_dotenv()
        
        # Initialize OpenAI model
        llm = ChatOpenAI(
            model="gpt-4",  # Changed from gpt-4o-mini which appears to be a typo
            temperature=0,
            timeout=60,  # Added reasonable timeout
            max_retries=2
        )
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        return llm, embeddings
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        st.error(f"Failed to initialize models: {str(e)}")
        return None, None

def process_message(
    query: str,
    llm: ChatOpenAI,
    index_name: str,
    embeddings: HuggingFaceEmbeddings,
    vectorstore: PineconeVectorStore,

) -> Tuple[str, List]:
    """Process the user message using the RAG system."""
    try:
        response, sources = RAG(
            query=query,
            llm=llm,
            index_name=index_name,
            embeddings=embeddings,
            vectorstore=vectorstore,
        )
        return response, sources
    except Exception as e:
        logger.error(f"Error in process_message: {str(e)}")
        return f"Error processing message: {str(e)}", []

def display_sources(sources: List) -> None:
    """Display sources in expandable sections with proper formatting."""
    if not sources:
        st.info("No sources available for this response.")
        return

    st.subheader("Sources")
    for i, doc in enumerate(sources, 1):
        try:
            with st.expander(f"Source {i}"):
                if hasattr(doc, 'page_content'):
                    st.markdown(f"**Content:** {doc.page_content}")
                    if hasattr(doc, 'metadata'):
                        for key, value in doc.metadata.items():
                            st.markdown(f"**{key.title()}:** {value}")
                else:
                    st.markdown(f"**Content:** {str(doc)}")
        except Exception as e:
            logger.error(f"Error displaying source {i}: {str(e)}")
            st.error(f"Error displaying source {i}")

def main():
    st.title("RAG Chatbot")
    
    INDEX_NAME = 'bpl-rag'

    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize models
    llm, embeddings = initialize_models()
    if not llm or not embeddings:
        st.error("Failed to initialize the application. Please check the logs.")
        return
    
    #initialize vectorstore
    pc = Pinecone(api_key=pinecone_api_key)
    
    index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Let Me Think..."):
                response, sources = process_message(
                    query=user_input,
                    llm=llm,
                    index_name=INDEX_NAME,
                    embeddings=embeddings,
                    vectorstore=vector_store
                )
                
                if isinstance(response, str):
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Display sources
                    display_sources(sources)
                else:
                    st.error("Received an invalid response format")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit + LangChain + OpenAI",
        help="An AI-powered chatbot with RAG capabilities"
    )

if __name__ == "__main__":
    main()