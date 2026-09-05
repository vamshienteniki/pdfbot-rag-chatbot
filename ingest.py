import os
import shutil
import warnings
from dotenv import load_dotenv

# Suppress non-critical deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


# Step 1: Load raw PDF(s)
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model


def main():
    print("Step 1: Loading PDF files...")
    documents = load_pdf_files(DATA_PATH)
    print(f"Loaded {len(documents)} document page(s).")

    print("Step 2: Creating text chunks...")
    text_chunks = create_chunks(documents)
    print(f"Created {len(text_chunks)} text chunk(s).")

    print("Step 3: Initializing embedding model...")
    embedding_model = get_embedding_model()

    print("Step 4: Storing embeddings in FAISS vector store...")
    if os.path.exists(DB_FAISS_PATH):
        shutil.rmtree(DB_FAISS_PATH)
        print(f"Cleared existing vector store at '{DB_FAISS_PATH}'.")

    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS vector store created and saved successfully at '{DB_FAISS_PATH}'.")


if __name__ == "__main__":
    main()