import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page Configuration
st.set_page_config(
    page_title="Gemini-Style RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Load environment variables (supports local .env or Streamlit Cloud Secrets)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")

# App Header
st.title("🤖 AI Document Chatbot")
st.caption("Created by Rana Usama - Chat with your PDFs using OpenRouter and local embeddings.")

# --- Sidebar for Document Management ---
st.sidebar.header("📁 Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload your PDF documents", 
    type=["pdf"], 
    accept_multiple_files=True
)

@st.cache_resource
def get_embeddings():
    """Load local HuggingFace embedding model once."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to build or update the vector store from uploaded files
def build_vector_store(files):
    all_chunks = []
    for file in files:
        # Save uploaded file to a temporary file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
        finally:
            os.remove(tmp_path)  # Clean up temp file

    if not all_chunks:
        return None

    # Create Chroma vector store in memory / temporary directory
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store

# Initialize Vector Store if files are uploaded
if uploaded_files:
    if "vector_store" not in st.session_state or st.sidebar.button("🔄 Refresh / Process Files"):
        with st.spinner("Processing documents into Vector DB..."):
            st.session_state.vector_store = build_vector_store(uploaded_files)
            st.sidebar.success(f"Successfully processed {len(uploaded_files)} file(s)!")
else:
    st.sidebar.info("👈 Please upload at least one PDF to start chatting.")
    st.session_state.pop("vector_store", None)

# --- Chat Interface Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input bar
user_query = st.chat_input("Ask a question about your document...")

if user_query:
    if "vector_store" not in st.session_state:
        st.warning("⚠️ Please upload and process a PDF document in the sidebar first.")
    else:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Setup retriever
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": 3}
                    )

                    # Initialize OpenRouter LLM
                    llm = ChatOpenAI(
                        model="openrouter/free", 
                        openai_api_key=OPENROUTER_API_KEY,
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=0.3,
                        default_headers={
                            "HTTP-Referer": "https://localhost",
                            "X-Title": "Streamlit RAG Chatbot",
                        }
                    )

                    # LCEL RAG Chain
                    template = (
                        "You are an assistant for question-answering tasks. "
                        "Use the following pieces of retrieved context to answer "
                        "the question. If you don't know the answer, say that you "
                        "don't know.\n\n"
                        "Context:\n{context}\n\n"
                        "Question: {input}"
                    )
                    prompt = ChatPromptTemplate.from_template(template)

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    rag_chain = (
                        {"context": retriever | format_docs, "input": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    answer = rag_chain.invoke(user_query)
                    st.markdown(answer)
                    
                    # Append assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"❌ An error occurred: {e}"
                    st.error(error_msg)
