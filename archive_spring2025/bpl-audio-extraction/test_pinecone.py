from dotenv import load_dotenv
import os
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")

# Print first few characters of API key to verify it's loaded
# (never print your full API key)
if api_key:
    print(f"API key loaded: {api_key[:5]}...")
else:
    print("API key not found!")

# Try to connect to Pinecone
try:
    pc = Pinecone(api_key=api_key)
    # List available indexes
    indexes = pc.list_indexes()
    print(f"Successfully connected to Pinecone! Available indexes: {indexes}")
except Exception as e:
    print(f"Connection failed: {e}")