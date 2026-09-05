import os
import time
import warnings
import streamlit as st
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

DB_FAISS_PATH = "vectorstore/db_faiss"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


@st.cache_resource
def get_vectorstore():
    if not os.path.exists(DB_FAISS_PATH):
        return None
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "input"])
    return prompt


def load_llm():
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment or .env file.")
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


def invoke_with_retry(qa_chain, user_input, max_retries=3, wait_seconds=6):
    """Invoke the QA chain with automatic retry on rate limit (429) errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return qa_chain.invoke({"input": user_input})
        except Exception as e:
            err_str = str(e)
            if "429" in err_str and attempt < max_retries:
                st.warning(f"⏳ Rate limited. Retrying in {wait_seconds}s... (Attempt {attempt}/{max_retries})")
                time.sleep(wait_seconds)
            else:
                raise


def main():
    st.title("🏥 MediBot - Medical Assistant")
    st.caption("Powered by OpenRouter · Free LLM")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context.

Context: {context}
Question: {input}

Start the answer directly. No small talk please."""

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Vector store not found. Please run 'python ingest.py' first.")
                return

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            llm = load_llm()
            custom_prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

            combine_docs_chain = create_stuff_documents_chain(llm, custom_prompt)
            qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

            with st.spinner("Thinking..."):
                response = invoke_with_retry(qa_chain, prompt)

            result = response["answer"]
            source_documents = response.get("context", [])

            # Format response with sources
            result_to_show = result + "\n\n---\n### 📚 Source Documents\n"
            for i, doc in enumerate(source_documents, 1):
                source_file = os.path.basename(doc.metadata.get("source", "PDF Document"))
                page_num = doc.metadata.get("page", None)
                page_str = f" (Page {page_num + 1})" if page_num is not None else ""
                snippet = doc.page_content.strip().replace("\n", " ")
                result_to_show += f"\n**Source {i}: `{source_file}`{page_str}**\n> {snippet}\n"

            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                st.error("⚠️ Model is rate limited. Please wait a few seconds and try again.")
            elif "401" in err_str:
                st.error("🔑 Invalid API key. Check your OPENROUTER_API_KEY in .env")
            elif "402" in err_str:
                st.error("💳 Insufficient credits. Switch to a free model in .env")
            else:
                st.error(f"Error: {err_str}")


if __name__ == "__main__":
    main()