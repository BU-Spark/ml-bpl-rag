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
from image_scraper import DigitalCommonwealthScraper
import shutil

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
        
        if "llm" not in st.session_state:
            # Initialize OpenAI model
            st.session_state.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Changed from gpt-4o-mini which appears to be a typo
                temperature=0,
                timeout=60,  # Added reasonable timeout
                max_retries=2
            )
        
        if "embeddings" not in st.session_state:
            # Initialize embeddings
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        if "pinecone" not in st.session_state:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            INDEX_NAME = 'bpl-rag'
            #initialize vectorstore
            pc = Pinecone(api_key=pinecone_api_key)
            
            index = pc.Index(INDEX_NAME)
            st.session_state.pinecone = PineconeVectorStore(index=index, embedding=st.session_state.embeddings)
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        st.error(f"Failed to initialize models: {str(e)}")
        return None, None

def process_message(
    query: str,
    llm: ChatOpenAI,
    vectorstore: PineconeVectorStore,

) -> Tuple[str, List]:
    """Process the user message using the RAG system."""
    try:
        response, sources = RAG(
            query=query,
            llm=llm,
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
                    st.markdown(f"**Content:** {doc.page_content[0:100] + ' ...'}")
                    if hasattr(doc, 'metadata'):
                        for key, value in doc.metadata.items():
                            st.markdown(f"**{key.title()}:** {value}")
                            
                        # Web Scraper to display images of sources
                        # Especially helpful if the sources are images themselves
                        # or are OCR'd text files
                        scraper = DigitalCommonwealthScraper()
                        images = scraper.extract_images(doc.metadata["URL"])
                        images = images[:1]
                        
                        # If there are no images then don't display them
                        if not images:
                                st.warning("No images found on the page.")
                                return
                                
                        # Download the images
                        # Delete the directory if it already exists
                        # to clear the existing cache of images for each listed source
                        output_dir = 'downloaded_images'
                        if os.path.exists(output_dir):
                            shutil.rmtree(output_dir)
                        
                        # Download the main image to a local directory
                        downloaded_files = scraper.download_images(images)
                
                        # Display the image using st.image
                        # Display the title of the image using img.get
                        st.image(downloaded_files, width=400, caption=[
                            img.get('alt', f'Image {i+1}') for i, img in enumerate(images)
                            ])

                else:
                    st.markdown(f"**Content:** {str(doc)}")
                    
        except Exception as e:
            logger.error(f"Error displaying source {i}: {str(e)}")
            st.error(f"Error displaying source {i}")

def main():
    st.title("Digital Commonwealth RAG")
    
    INDEX_NAME = 'bpl-rag'

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize models
    initialize_models()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your query here...")
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... Please be patient, I'm a little slow right now..."):
                response, sources = process_message(
                    query=user_input,
                    llm=st.session_state.llm,
                    vectorstore=st.session_state.pinecone
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
        "Built with Langchain + Streamlit + Pinecone",
        help="Natural Language Querying for Digital Commonwealth"
    )
    st.markdown("The Digital Commonwealth site provides access to photographs, manuscripts, books, audio recordings, and other materials of historical interest that have been\ndigitized and made available by members of Digital Commonwealth, a statewide consortium of libraries, museums, archives, and historical societies from across Massachusetts.")

if __name__ == "__main__":
    main()