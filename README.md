# 📊 Financial Report Analyzer

### 🔗 [Live App → financial-report-analyzer-rag.streamlit.app](https://financial-report-analyzer-rag.streamlit.app/)

AI-powered financial document analysis system using RAG (Retrieval Augmented Generation).  
Analyzes SEC filings (10-K / 20-F) from 7 IT/consulting companies.

---

## Companies Covered

| Ticker | Company | Filing Type |
|--------|---------|-------------|
| MSFT | Microsoft | 10-K |
| GOOGL | Alphabet (Google) | 10-K |
| ACN | Accenture | 10-K |
| IBM | IBM | 10-K |
| CTSH | Cognizant | 10-K |
| INFY | Infosys | 20-F |
| WIT | Wipro | 20-F |

## Tech Stack

- **Document Source:** SEC EDGAR (10-K / 20-F filings)
- **Parsing:** BeautifulSoup (HTML extraction from SGML submissions)
- **Chunking:** LangChain RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Embeddings:** all-MiniLM-L6-v2 (via ChromaDB default)
- **Vector DB:** ChromaDB (persistent, cosine similarity, 5,786 chunks)
- **LLM:** Llama 3.3 70B via Groq API (free tier)
- **Agent:** LangGraph ReAct agent with financial tools
- **UI:** Streamlit
- **Deployment:** Streamlit Community Cloud

## Project Structure

```
├── app.py                              # Streamlit UI (main application)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── .gitignore                          # Excludes secrets and heavy folders
├── Step1_Download_Filings.ipynb        # Download 10-K/20-F filings from SEC EDGAR
├── Step2_Parse_and_Extract.ipynb       # Parse HTML from full-submission.txt, extract text
├── Step3_Chunking_and_Embedding.ipynb  # Chunk text & store in ChromaDB
├── Step4_RAG_Pipeline.ipynb            # RAG Q&A pipeline with Groq
├── Step5_Agent_and_Tools.ipynb         # LangGraph agent with financial tools
└── data/
    └── chroma_db/                      # Vector database (included in repo)
```

**Not included in repo (can be regenerated):**
- `data/sec_filings/` — Raw downloaded filings (re-run Step 1 to download)
- `data/extracted_texts/` — Parsed text files (re-run Step 2 to regenerate)

## Architecture

```
User Question
     ↓
  [Embedding] → query vector (all-MiniLM-L6-v2)
     ↓
  [ChromaDB] → top-k relevant chunks (cosine similarity)
     ↓
  [Prompt Construction] → system prompt + context + question
     ↓
  [Groq / Llama 3.3 70B] → grounded answer
     ↓
  [Streamlit UI] → display with source citations
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create a `.streamlit/secrets.toml` file in the project root:
```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

Get a free Groq API key at: https://console.groq.com

## Groq API Key — Where to Update

### In the Notebooks (Step 4, Step 5, and Step 6)

If you want to re-run the notebooks, paste your Groq API key in these cells:

- **Step4_RAG_Pipeline.ipynb** → Cell 4.2:
  ```python
  GROQ_API_KEY = "your-groq-api-key-here"
  ```

- **Step5_Agent_and_Tools.ipynb** → Cell 5.2:
  ```python
  os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"
  ```
  ```python
  GROQ_API_KEY = "your-groq-api-key-here"
  ```

### In Streamlit Cloud (for the deployed app)

If your API key is exhausted or you want to rotate it:

1. Go to https://share.streamlit.io
2. Find your app and click the **three dots (⋮)** menu
3. Click **"Settings"** → **"Secrets"** tab
4. Update the key:
   ```
   GROQ_API_KEY = "your-new-groq-api-key-here"
   ```
5. Click **"Save"** — the app will reboot automatically

### In Local Development

Update the key in `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-new-groq-api-key-here"
```

### Free Tier Limits (Groq)

- **llama-3.3-70b-versatile:** 100K tokens/day, 30 requests/min
- **llama-3.1-8b-instant:** higher limits, suitable for testing
- Limits reset daily at midnight Pacific Time (12:30 PM IST next day)
- Get a new key anytime at https://console.groq.com

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (include the `data/chroma_db/` directory)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → select your repo → set main file to `app.py`
4. In "Advanced settings" → "Secrets", add:
   ```
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
5. Click "Deploy"
