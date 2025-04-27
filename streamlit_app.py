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
    page_title="Digital Commonwealth",
    page_icon="📚",
    layout="wide"
)

# Custom CSS to match Digital Commonwealth styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1A2A57;
        font-family: 'Lora', serif;
    }
    h2, h3 {
        color: #1A2A57;
        font-family: 'Lora', serif;
        margin-top: 20px;
    }
    .stButton button {
        background-color: #1A2A57;
        color: white;
        border-radius: 4px;
    }
    .stTextInput>div>div>input {
        border-radius: 4px;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    footer {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
    .css-1q8dd3e {
        padding: 2rem 1rem 1.5rem;
    }
    .css-18e3th9 {
        padding-top: 0;
    }
    .sources-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
    }
    .source-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        background-color: white;
    }
    .search-bar {
        background-color: #1A2A57;
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: white;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .logo {
        margin-right: 20px;
    }
    .browse-section {
        margin: 30px 0;
        padding: 20px;
        background-color: #e9ecef;
        border-radius: 4px;
    }
    .collections-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 15px;
    }
    .collection-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        border: 1px solid #ddd;
        transition: transform 0.2s;
    }
    .collection-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_models() -> Tuple[Optional[ChatOpenAI], HuggingFaceEmbeddings]:
    """Initialize the language model and embeddings."""
    try:
        load_dotenv()
        
        if "llm" not in st.session_state:
            # Initialize OpenAI model
            st.session_state.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                timeout=60,
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
    """Display sources in a grid layout similar to Digital Commonwealth."""
    if not sources:
        st.info("No sources available for this response.")
        return

    st.markdown("### Sources")
    
    # Start a container for the grid
    st.markdown('<div class="sources-container">', unsafe_allow_html=True)
    
    for i, doc in enumerate(sources, 1):
        try:
            st.markdown(f'<div class="source-card">', unsafe_allow_html=True)
            
            # Process content and metadata
            if hasattr(doc, 'page_content'):
                st.markdown(f"<h4>Source {i}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Content:</strong> {doc.page_content[0:100] + '...'}</p>", unsafe_allow_html=True)
                
                if hasattr(doc, 'metadata'):
                    for key, value in doc.metadata.items():
                        st.markdown(f"<p><strong>{key.title()}:</strong> {value}</p>", unsafe_allow_html=True)
                    
                    # Web Scraper to display images of sources
                    scraper = DigitalCommonwealthScraper()
                    images = scraper.extract_images(doc.metadata["URL"])
                    images = images[:1]
                    
                    # If there are images, display them
                    if images:
                        # Download the images
                        output_dir = 'downloaded_images'
                        if os.path.exists(output_dir):
                            shutil.rmtree(output_dir)
                        
                        # Download the main image to a local directory
                        downloaded_files = scraper.download_images(images)
                
                        # Display the image
                        st.image(downloaded_files, width=250, caption=[
                            img.get('alt', f'Image {i+1}') for i, img in enumerate(images)
                        ])
            else:
                st.markdown(f"<p><strong>Content:</strong> {str(doc)}</p>", unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
                    
        except Exception as e:
            logger.error(f"Error displaying source {i}: {str(e)}")
            st.error(f"Error displaying source {i}")
    
    # Close the grid container
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header with BPL/Digital Commonwealth styling
    st.markdown("""
    <div class="header-container">
        <div class="logo">
            <img src="https://www.bpl.org/wp-content/themes/bpl-iwh/images/bpl-logo.svg" width="80">
        </div>
        <div>
            <h1>Digital Commonwealth</h1>
            <p>An online library of historical materials from libraries, museums, archives, and historical societies across Massachusetts</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize models
    initialize_models()

    # Main search interface
    st.markdown('<div class="search-bar">', unsafe_allow_html=True)
    st.markdown("<h3>Search Collections</h3>", unsafe_allow_html=True)
    
    # Chat input tabs
    tab1, tab2 = st.tabs(["Basic Search", "Chat with Collections"])
    
    with tab1:
        basic_query = st.text_input("Search for materials by keyword, subject, location or format", key="basic_query")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.selectbox("Format", ["All Formats", "Photographs", "Maps", "Manuscripts", "Books", "Audio"])
        with col2:
            st.selectbox("Collection", ["All Collections", "BPL", "Archives", "Museums", "Historical Societies"])
        with col3:
            st.selectbox("Date Range", ["All Dates", "Before 1900", "1900-1950", "1950-2000", "After 2000"])
        if st.button("Search", key="basic_search"):
            if basic_query:
                with st.spinner("Searching collections..."):
                    response, sources = process_message(
                        query=basic_query,
                        llm=st.session_state.llm,
                        vectorstore=st.session_state.pinecone
                    )
                    
                    st.markdown("### Results")
                    st.markdown(response)
                    
                    # Display sources in Digital Commonwealth style
                    display_sources(sources)
    
    with tab2:
        # Display chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask a question about Massachusetts history and collections...")
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching through historical collections..."):
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Browse collections section
    st.markdown('<div class="browse-section">', unsafe_allow_html=True)
    st.markdown("## Browse Collections")
    st.markdown("Explore curated collections of historic materials grouped by topic, date, location, or format.")
    
    # Sample collections grid
    st.markdown('<div class="collections-grid">', unsafe_allow_html=True)
    
    collections = [
        {"title": "Photographs", "icon": "📷", "count": "150,000+"},
        {"title": "Maps", "icon": "🗺️", "count": "8,500+"},
        {"title": "Manuscripts", "icon": "📜", "count": "25,000+"},
        {"title": "Books", "icon": "📚", "count": "80,000+"},
        {"title": "Audio", "icon": "🔊", "count": "3,000+"},
        {"title": "Boston History", "icon": "🏙️", "count": "45,000+"},
        {"title": "Natural History", "icon": "🌿", "count": "12,000+"},
        {"title": "Art & Architecture", "icon": "🏛️", "count": "18,000+"}
    ]
    
    for collection in collections:
        st.markdown(f"""
        <div class="collection-card">
            <h3>{collection["icon"]} {collection["title"]}</h3>
            <p>{collection["count"]} items</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 20px 0;">
        <div>
            <p>Digital Commonwealth is supported by the <strong>Boston Public Library</strong> through the Library for the Commonwealth program.</p>
            <p>The program provides access to photographs, manuscripts, books, audio recordings, and other materials of historical interest.</p>
        </div>
        <div>
            <p>Built with Langchain + Streamlit + Pinecone</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()