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
from RAG_cloudsql_vector import CloudSQLVectorStore
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
                model_name="sentence-transformers/all-mpnet-base-v2"
                #model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        if "pinecone" not in st.session_state:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            INDEX_NAME = 'bpl-test'
            #initialize vectorstore
            pc = Pinecone(api_key=pinecone_api_key)
            
            index = pc.Index(INDEX_NAME)
            st.session_state.pinecone = PineconeVectorStore(index=index, embedding=st.session_state.embeddings)
        
        if "vectorstore" not in st.session_state:
            #st.session_state.vectorstore = CloudSQLVectorStore(embedding=st.session_state.embeddings)
            st.session_state.vectorstore = st.session_state.pinecone
        
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
    """Display sources with minimal output: content preview, source, URL, and image/audio if available."""
    if not sources:
        st.info("No sources available for this response.")
        return

    st.subheader("Sources")
    for doc in sources:
        try:
            metadata = doc.metadata
            source = metadata.get("source", "Unknown Source")
            title = metadata.get("title_info_primary_tsi", "Unknown Title")
            format_type = metadata.get("format", "").lower()

            is_audio = "audio" in format_type

            expander_title = f"🔊 {title}" if is_audio else title

            with st.expander(expander_title):
                # Content preview
                if hasattr(doc, 'page_content'):
                    st.markdown(f"**Content:** {doc.page_content[:300]} ...")

                # URL building
                doc_url = metadata.get("URL", "").strip()
                if not doc_url and source:
                    doc_url = f"https://www.digitalcommonwealth.org/search/{source}"

                st.markdown(f"**Source ID:** {source}")
                st.markdown(f"**Format:** {format_type if format_type else 'Not specified'}")
                st.markdown(f"**URL:** {doc_url}")

                # 🔊 Try to show audio if it's an audio entry and there's a media file
                if is_audio:
                    # Try to find a playable media file — if metadata has audio URLs
                    # For now, just embed a dummy player or placeholder
                    st.info("This is an audio entry.")
                    # Optionally:
                    # st.audio("https://example.com/audio-file.mp3")  # replace with real audio URL
                else:
                    # 🖼️ Show image if it's not audio
                    scraper = DigitalCommonwealthScraper()
                    images = scraper.extract_images(doc_url)
                    images = images[:1]

                    if images:
                        output_dir = 'downloaded_images'
                        if os.path.exists(output_dir):
                            shutil.rmtree(output_dir)
                        downloaded_files = scraper.download_images(images)
                        st.image(downloaded_files, width=400, caption=[
                            img.get('alt', f'Image') for img in images
                        ])
        except Exception as e:
            logger.warning(f"[display_sources] Error displaying document: {e}")
            st.error("Error displaying one of the sources.")



def main():
    st.title("Digital Commonwealth RAG 🤖")

    INDEX_NAME = 'bpl-rag'

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False

    if "num_sources" not in st.session_state:
        st.session_state.num_sources = 10
        

    initialize_models()

    # 🔵 Settings button
    open_settings = st.button("⚙️ Settings")

    if open_settings:
        st.session_state.show_settings = True

    if st.session_state.show_settings:
        with st.container():
            st.markdown("---")
            st.markdown("### ⚙️ Settings")

            num_sources = st.number_input(
                "Number of Sources to Display",
                min_value=1,
                max_value=100,
                value=st.session_state.num_sources,
                step=1,
            )
            st.session_state.num_sources = num_sources

            close_settings = st.button("❌ Close Settings")
            if close_settings:
                st.session_state.show_settings = False
            st.markdown("---")

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ⬇️ CHAT INPUT BOX always stuck to bottom
    user_input = st.chat_input("Type your question here...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking... Please be patient..."):
                response, sources = process_message(
                    query=user_input,
                    llm=st.session_state.llm,
                    vectorstore=st.session_state.vectorstore
                )

                if isinstance(response, str):
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                    display_sources(sources[:int(st.session_state.num_sources)])
                else:
                    st.error("Received an invalid response format")

    # Footer (optional, will be above chat input)
    st.markdown("---")
    st.markdown(
        "Built with Langchain + Streamlit + Pinecone",
        help="Natural Language Querying for Digital Commonwealth"
    )
    st.markdown(
        "The Digital Commonwealth site provides access to photographs, manuscripts, books, "
        "audio recordings, and other materials of historical interest that have been digitized "
        "and made available by members of Digital Commonwealth."
    )

if __name__ == "__main__":
    main()