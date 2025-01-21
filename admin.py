import os
import streamlit as st
import subprocess
import tempfile
import shutil
from uuid import uuid4
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from unstructured.partition.docx import partition_docx
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("API key not found. Please check your .env file.")
    st.stop()

# Helper functions
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def load_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    for idx, doc in enumerate(documents):
        doc.metadata["source"] = file.name
        doc.metadata["page_number"] = idx + 1
    os.remove(temp_file_path)
    return documents

def load_docx(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    elements = partition_docx(filename=temp_file_path)
    full_text = "\n".join(element.text for element in elements if element.text)
    documents = []
    chunks = text_splitter.split_text(full_text)
    for idx, chunk in enumerate(chunks):
        metadata = {"source": file.name, "page_number": idx + 1, "section": "General Section"}
        documents.append(Document(page_content=chunk, metadata=metadata))
    os.remove(temp_file_path)
    return documents

def add_documents_with_ids(vectorstore, documents, batch_size=5000):
    total_docs = len(documents)
    for doc in documents:
        unique_id = str(uuid4())
        doc.metadata["id"] = unique_id
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        vectorstore.add_documents(batch)

def create_new_collection(collection_name):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    collection_path = os.path.join("./chroma_db", collection_name)
    os.makedirs(collection_path, exist_ok=True)
    return Chroma(embedding_function=embedding_function, persist_directory=collection_path)

def load_collection(collection_name):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    collection_path = os.path.join("./chroma_db", collection_name)
    if not os.path.exists(collection_path):
        raise ValueError(f"Collection '{collection_name}' does not exist.")
    return Chroma(embedding_function=embedding_function, persist_directory=collection_path)

def delete_collection(collection_name):
    import time
    collection_path = os.path.join("./chroma_db", collection_name)
    if os.path.exists(collection_path):
        try:
            vectorstore = load_collection(collection_name)
            vectorstore.persist()
            del vectorstore
            time.sleep(1)
            shutil.rmtree(collection_path)
            st.success(f"Collection '{collection_name}' deleted successfully.")
        except Exception as e:
            st.error(f"Failed to delete collection '{collection_name}': {str(e)}")
    else:
        st.warning(f"Collection '{collection_name}' does not exist.")

def delete_document(collection_name, document_name):
    vectorstore = load_collection(collection_name)
    all_docs = vectorstore.get()
    document_ids_to_delete = [
        all_docs["ids"][idx] for idx, metadata in enumerate(all_docs["metadatas"])
        if metadata.get("source") == document_name
    ]
    if document_ids_to_delete:
        vectorstore.delete(ids=document_ids_to_delete)

# Streamlit UI for the Admin Page
st.title("Admin Page: Manage Database")

# File Upload Section
st.subheader("Upload Files for Ingestion")
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# Collection Selection or Creation
st.subheader("Select or Create a Collection")
base_path = "./chroma_db/"
existing_collections = [
    name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))
]

collection_option = st.radio(
    "Choose an option:",
    ("Select an existing collection", "Create a new collection"),
)

if collection_option == "Select an existing collection":
    if existing_collections:
        selected_collection = st.selectbox("Select a collection", existing_collections)
    else:
        st.info("No collections available. Create a new one.")
        selected_collection = None
else:  # Create a new collection
    new_collection_name = st.text_input("Enter a name for the new collection")
    selected_collection = new_collection_name.strip() if new_collection_name else None

if st.button("Ingest Files"):
    if not selected_collection:
        st.error("Please select or create a collection.")
    elif not uploaded_files:
        st.error("Please upload at least one file.")
    else:
        try:
            collection_name = selected_collection
            if collection_option == "Create a new collection":
                vectorstore = create_new_collection(collection_name)
                st.success(f"Created new collection '{collection_name}'.")
            else:
                vectorstore = load_collection(collection_name)

            documents = []
            for file in uploaded_files:
                if file.name.endswith(".pdf"):
                    documents.extend(load_pdf(file))
                elif file.name.endswith(".docx"):
                    documents.extend(load_docx(file))
            add_documents_with_ids(vectorstore, documents)
            st.success(f"Ingested {len(documents)} documents into collection '{collection_name}'.")
        except ValueError as e:
            st.error(str(e))
            

# Chatbot Button
if st.button("Open Chatbot"):
    # Run the chatbot script in a separate process
    subprocess.run(["streamlit", "run", "rag.py"])  # Replace 'chatbot.py' with your chatbot script path

# Collection Management
st.subheader("Select Collection")
if existing_collections:
    selected_collection = st.selectbox("Select a collection to manage", existing_collections)

    # if st.button("Delete Collection"):
    #     delete_collection(selected_collection)
    #     st.warning("Please refresh the app to see updated collections.")

    st.markdown("Manage Documents:")
    try:
        vectorstore = load_collection(selected_collection)
        all_docs = vectorstore.get()
        document_names = {metadata.get("source") for metadata in all_docs.get("metadatas", [])}

        if document_names:
            selected_document = st.selectbox("Select a document to delete", list(document_names))
            if st.button("Delete Document"):
                delete_document(selected_collection, selected_document)
                st.success(f"Document '{selected_document}' deleted successfully.")
        else:
            st.info("No documents available in the selected collection.")
    except ValueError as e:
        st.error(str(e))
else:
    st.info("No collections available.")
