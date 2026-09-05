import os
import warnings
from dotenv import load_dotenv

# Suppress non-critical deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Step 1: Setup LLM (OpenRouter)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def load_llm():
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment or .env file.")
    # Use a free model on OpenRouter (no credits required)
    model_name = os.environ.get("OPENROUTER_MODEL", "minimax/minimax-m3:free")
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.5,
        max_tokens=512,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "MediBot RAG",
        }
    )
    return llm




# Step 2: Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {input}

Start the answer directly. No small talk please.
"""


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "input"]
    )
    return prompt


def main():
    DB_FAISS_PATH = "vectorstore/db_faiss"
    if not os.path.exists(DB_FAISS_PATH):
        print(f"Error: Vector store not found at '{DB_FAISS_PATH}'. Please run 'python ingest.py' first.")

        return

    # Load Database
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={'k': 3})
    llm = load_llm()
    prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

    # Modern LangChain LCEL retrieval chain creation
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    user_query = input("Write Query Here: ")
    if user_query.strip():
        response = qa_chain.invoke({'input': user_query})
        print("\n" + "=" * 60)
        print("💡 ANSWER:\n")
        print(response["answer"])
        print("=" * 60)
        print("\n📚 SOURCE DOCUMENTS:")
        for i, doc in enumerate(response.get("context", []), 1):
            source_file = os.path.basename(doc.metadata.get("source", "PDF Document"))
            page_num = doc.metadata.get("page", None)
            page_str = f" (Page {page_num + 1})" if page_num is not None else ""
            print(f"\n--- Document {i}: {source_file}{page_str} ---")
            print(doc.page_content.strip())
        print("=" * 60)



if __name__ == "__main__":
    main()