"""
Study Notes Q&A Assistant — LangChain + Supabase RAG
------------------------------------------------------
Upload your notes/PDFs once; they're stored persistently in a
Supabase (Postgres + pgvector) database. Ask questions anytime,
even after restarting the app — unlike the in-memory version.

Run with: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from tesr_supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.documents import Document

load_dotenv()  # Load environment variables from .env file

st.set_page_config(
    page_title="Study Notes Q&A Assistant", page_icon=":books:", layout="centered"
)
st.title("Study Notes Q&A Assistant :books:")
st.caption(
    "Upload your notes/PDFs once; they're stored persistently in a Supabase (Postgres + pgvector) database. Ask questions anytime, even after restarting the app — unlike the in-memory version."
)

# ---------------------------------------------------------
# CONNECTIONS (cached so they're only set up once)
# ---------------------------------------------------------


@st.cache_resource
def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)


@st.cache_resource
def get_embeddings():
    # Free , runs locally - no API cost, no extra key needed
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def get_vector_store():
    return SupabaseVectorStore(
        client=get_supabase_client(),
        embedding=get_embeddings(),
        table_name="documents",
        query_name="match_documents",
    )


def get_llm(model_name, temperature):
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model=model_name,
        temperature=temperature,
    )


# ---------------------------------------------------------
# SIDEBAR: settings + document upload
# ---------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.1)

    st.divider()
    st.subheader("📄 Add notes to your knowledge base")
    uploaded_files = st.file_uploader(
        "Upload a PDF", type=["pdf", "txt", "docx"], accept_multiple_files=True
    )

    if uploaded_files is not None and st.button("Index this document"):
        total_chunks = 0
        for uploaded_file in uploaded_files:
            with st.spinner("Splitting and embedding document"):
                # Save temporarily so PyPDFLoader (LangChain's PDF reader) can open it
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # LangChain's loader + splitter replace your old chunk_text() function
            loader = PyMuPDFLoader(temp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(pages)

            # This single call embeds every chunk AND uploads it to Supabase —
            # replaces your manual embedder.encode() + session_state storage.
            vector_store = get_vector_store()
            vector_store.add_documents(chunks)

            os.remove(temp_path)  # Clean up the temporary file
            total_chunks += len(chunks)
        st.success(
            f"Indexed {total_chunks} chunks from {len(uploaded_files)} file(s) into Supabase."
        )


# ---------------------------------------------------------
# CHAT
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something about your uploaded notes...")

if user_input:
    if not os.getenv("GROQ_API_KEY"):
        st.error(
            "Please set your GROQ_API_KEY in the .env file to use this app. You can get a free key from https://groq.com/"
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            # This one call replaces your manual cosine similarity function —
            # LangChain + Supabase handle the vector search for you.
            vector_store = get_vector_store()
            relevant_docs = vector_store.similarity_search(user_input, k=3)
            context = "\n\n--\n\n".join(doc.page_content for doc in relevant_docs)
            prompt = (
                "Use the following notes excerpts to answer the question. "
                "If the answer isn't in the excerpts, say so.\n\n"
                f"Excerpts:\n{context}\n\n"
                f"Question: {user_input}"
            )

            llm = get_llm(model, temperature)
            response = llm.invoke(prompt)
            answer = response.content

        st.markdown(answer)

        if relevant_docs:
            with st.expander("📎 Sources used"):
                for i, doc in enumerate(relevant_docs, 1):
                    st.caption(f"Excerpt {i}: {doc.page_content[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
