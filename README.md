# 📄 PDFBot — RAG-based Document Chatbot

A RAG (Retrieval-Augmented Generation) based chatbot built with LangChain, FAISS, and Streamlit to answer questions from uploaded PDF documents using free LLMs via OpenRouter.

---

## 🚀 Features

- 📄 Loads and processes PDF documents with automated text chunking and embedding generation
- 🔍 Efficient information retrieval using FAISS vector search with HuggingFace embeddings
- 🤖 Context-aware responses powered by MiniMax M3 LLM via OpenRouter API
- 💬 Interactive chat UI built with Streamlit
- 📚 Shows source documents with page references for every answer

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| LLM | MiniMax M3 (via OpenRouter — Free) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Framework | LangChain |
| UI | Streamlit |
| Loader | PyPDFLoader |

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/medibot.git
cd medibot
```

### 2. Create virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY="your_openrouter_api_key_here"
OPENROUTER_MODEL="minimax/minimax-m3:free"
```
> Get your free API key at: https://openrouter.ai/keys

### 5. Add your PDF files
Place your medical PDF files inside the `data/` folder.

### 6. Run ingestion (first time only)
```bash
python ingest.py
```

### 7. Launch the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
medibot/
├── data/               # Put your PDF files here
├── vectorstore/        # Auto-generated FAISS index (after ingest.py)
├── app.py              # Streamlit chat UI
├── ingest.py           # PDF loader + embedding pipeline
├── query.py            # CLI query interface
├── requirements.txt    # Python dependencies
└── .env                # API keys (not pushed to GitHub)
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter API key (required) |
| `OPENROUTER_MODEL` | Model to use (default: `minimax/minimax-m3:free`) |

---

## 📝 Notes

- Run `python ingest.py` whenever you add new PDFs to the `data/` folder
- The `vectorstore/` folder is excluded from git — regenerate it locally
- Use any free model from [openrouter.ai/models](https://openrouter.ai/models?q=free) by updating `OPENROUTER_MODEL` in `.env`
