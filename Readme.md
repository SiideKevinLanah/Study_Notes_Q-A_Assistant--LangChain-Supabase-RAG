# Study Notes Q&A Assistant - LangChain + Supabase RAG

A RAG chatbot that answers questions about your uploaded notes/PDFs, using a **production-grade** : LangChain (industry-standard LLM framework) and Supabase (Postgres + pgvector) for presistent vector storage - instead of the in-memory , hand-built version from an earlier project.

## Why this stack
- **LangChain** replaces hand-written chunking , embedding , and prompt-building code with standard, reusable components - the same library most real LLM application use.
- **Supabase (pgvector)** relpaces in-memory session storage with a real, presisitent database. Your notes stay searchable across restarts, deployments, and even different devices.

## Features
- **Multi-file upload** - index several several PDFs in one batch ,not just one at a time
- **Resilient indexing** — if one PDF is corrupted or malformed, it's skipped with a clear error message instead of crashing the whole upload; the rest still get indexed
- **PyMuPDF-based PDF parsing** — more tolerant of unusual or non-standard PDFs than basic PDF readers
- **Persistent vector storage** — documents live in Supabase (Postgres + pgvector), not in memory, so they survive app restarts
- **Source transparency** — every answer includes an expandable "Sources used" section showing exactly which excerpts the AI pulled from

## Setup 

### 1. Create a free Supabase project 
Go to supabaase.com, sign up , create a new project. Note your **Project URL** and **service_role API key** (Settings → API — use `service_role`, not `anon`, since this app runs locally/server-side, never in a browser).

### 2. Set up the database
In your Supabase dashboard, go to **SQL Editor → New Query**, paste the contents of `supabase_setup.sql`, and run it. This:
- Enables the `pgvector` extension
- Creates a `documents` table to store text chunks + their embeddings
- Creates a `match_documents` function that LangChain uses to run similarity search

### 3. Get a free Groq API key
console.groq.com/keys

### 4. Create your `.env` file
API_KEY=your_groq_api_key_here
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_supabase_service_role_key_here

### 5. Install dependencies
pip install -r requirements.txt

### 6. (Optional) Verify your Supabase connection before running the full app
python3 test_supabase.py

Should print `Success! Connection works. Result: []` if everything's set up correctly. Useful for isolating connection problems from app-level bugs.

### 7. Run the app
streamlit run app.py

## How it works (for your report)

### Indexing (uploading notes)
1. User selects one or more PDFs (`accept_multiple_files=True`)
2. Each file is looped through individually — saved temporarily to disk, loaded with `PyMuPDFLoader`, split into chunks with `RecursiveCharacterTextSplitter`
3. `vector_store.add_documents(chunks)` embeds every chunk (via `HuggingFaceEmbeddings`, running locally) AND uploads it to Supabase in one call
4. If a specific file fails (e.g. corrupted PDF), it's caught, logged, and skipped — the loop continues with the remaining files
5. A summary reports how many chunks were indexed and from how many files, plus which files (if any) failed

### Answering questions
1. `vector_store.similarity_search(query, k=3)` sends the question to Supabase's `match_documents` SQL function, which performs cosine similarity search inside the database
2. Retrieved chunks are combined into a single prompt along with the question
3. `ChatGroq` (LangChain's wrapper around the Groq API) generates the answer
4. Retrieved sources are shown in a collapsible expander for transparency

## Key differences from an in-memory RAG implementation (worth highlighting in your report)

| | In-memory version | This project |
|---|---|---|
| Chunking | Custom function | LangChain's `RecursiveCharacterTextSplitter` |
| Embedding storage | In-memory, lost on restart | Supabase/Postgres, persistent |
| Similarity search | Manual NumPy cosine similarity | SQL function inside the database |
| PDF parsing | Basic reader, fails on malformed files | PyMuPDF, more tolerant + wrapped in error handling |
| Multi-document support | One file at a time | Batch upload with per-file failure isolation |
| Framework | Hand-built | LangChain (industry standard) |

## Troubleshooting

- **`Invalid API key` from Supabase** — double-check you copied the full `service_role` key (they're long, 200+ characters) with no missing characters or extra whitespace; run `test_supabase.py` to confirm in isolation
- **`PdfStreamError: Stream has ended unexpectedly`** — the specific PDF file is genuinely corrupted/incomplete; try re-downloading or re-exporting it. The app will skip this file and continue with others rather than crashing.
- **API key errors on chat** — make sure `API_KEY` (Groq) is set consistently — the same variable name must be used everywhere it's read in `app.py`

## Deploying it

Streamlit Community Cloud is the simplest free option:
1. Push the project to a GitHub repo (`.gitignore` excludes `.env`)
2. Connect the repo at share.streamlit.io
3. Add your API keys under the app's "Secrets" settings
4. You get a public URL to link in your report

## Suggested report additions

- Explain why moving from in-memory to a real vector database matters for production systems (multi-user support, persistence, scale)
- Discuss the resilience design (per-file error handling) as a real engineering consideration, not just a nice-to-have
- Compare this against an in-memory RAG implementation directly — shows growth and deeper understanding

