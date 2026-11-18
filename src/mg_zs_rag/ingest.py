import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import BSHTMLLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mg_zs_rag.config import (
    CHROMA_DIR,
    CHUNK_OUVERLAP,
    CHUNK_SIZE,
    HTML_PATH,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_EMBEDDING_MODEL,
    check_ingest_config,
)


def main():
    # 1. check congiguration
    check_ingest_config()

    # 2.load HTML document
    print("Loading MG ZS manual...")

    loader = BSHTMLLoader(
        file_path = str(HTML_PATH)
    )

    documents = loader.load()
    print(f'loaded{len(documents)}')


    # 3. split documents
    print("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OUVERLAP,
        length_function = len,
        add_start_index = True
    )

    splits = text_splitter.split_documents(documents)


    # 4. Initialize NVIDIA embeddings

    embeddings = NVIDIAEmbeddings(
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        model=NVIDIA_EMBEDDING_MODEL,
        truncate='END'
    )

    # 5. rebuild vector database

    if CHROMA_DIR.exists():
        print("Remove existing Chroma database...")
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True,exist_ok=True)

    # 6. store embeddings in Chroma

    print("Creating Chroma vector database...")
    vectorstore = Chroma.from_documents(
        documents = splits,
        embedding=embeddings,
        collection_name="mg_zs_manual",
        persist_directory=str(CHROMA_DIR)
    )

    print("\n knowledge base created successfully")

    print(f'Stored chunks:{vectorstore._collection.count()}')

    print(f'Database path: {CHROMA_DIR}')

if __name__ == '__main__':
    main()

