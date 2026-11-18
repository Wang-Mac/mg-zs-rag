import os
from pathlib import Path

from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#.env
load_dotenv(PROJECT_ROOT / '.env')

#Data Path
DATA_DIR = PROJECT_ROOT / 'data'

RAW_DATA_DIR = DATA_DIR / 'raw'

HTML_PATH = RAW_DATA_DIR / 'mg-zs-warning-messages.html'

CHROMA_DIR = DATA_DIR / 'chroma'

#Nvidia Api
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
NVIDIA_BASE_URL = os.getenv('NVIDIA_BASE_URL',"https://integrate.api.nvidia.com/v1")

#Nvidia Model
NVIDIA_LLM_MODEL = os.getenv(
    'NVIDIA_LLM_MODEL',
    'nvidia/nemotron-3.5-lightning-30b-a3b'
)

NVIDIA_EMBEDDING_MODEL = os.getenv(
    'NVIDIA_EMBEDDING_MODEL',
    'nvidia/nemotron-3.5-embed-1b'
)

#RAG
CHUNK_SIZE = 1000
CHUNK_OUVERLAP = 200
RETRIEVER_K = 4

#check
def check_api_config():
    '''Check NVIDIA API Configuration'''
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY is missing. Check your .env file.")


def check_ingest_config():
    """Check configuration before building the knowledge base."""
    check_api_config()
    if not HTML_PATH.exists():
        raise FileNotFoundError(f"HTML file not found: {HTML_PATH}")


def check_chat_config():
    """Check configuration before starting the chatbot."""
    check_api_config()
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "Chroma database not found. "
            "Please run ingest.py first."
        )


    



