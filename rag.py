import os
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# # Set expiration date and time
# EXPIRATION_DATETIME = datetime(2024, 12, 31, 23, 59, 59)  # Example: Dec 31, 2024, 11:59:59 PM

# # Get the current date and time
# current_datetime = datetime.now()

# # Check if the application has expired
# if current_datetime > EXPIRATION_DATETIME:
#     st.error("This chatbot is no longer available. Please contact support.")
#     st.stop()

# # Calculate remaining time
# remaining_time = EXPIRATION_DATETIME - current_datetime

# # Extract days, hours, minutes, and seconds
# days = remaining_time.days
# hours, remainder = divmod(remaining_time.seconds, 3600)
# minutes, seconds = divmod(remainder, 60)


# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("API key not found. Please check your .env file.")
    st.stop()

# Rest of your chatbot code...


# Helper function: Retrieve collections
def get_collections():
    collection_names = [name for name in os.listdir('./chroma_db/') if os.path.isdir(f"./chroma_db/{name}")]
    if not collection_names:
        st.error("No collections found. Please add a collection first.")
        st.stop()
    return collection_names

# Configure vector database
def get_vector_db(selected_collection):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=f"./chroma_db/{selected_collection}", embedding_function=embedding_function)

# Generate prompt for RAG
def generate_rag_prompt(query, context, metadata):
    metadata_str = "\n".join(
        [
            f"Source: {meta['source']} | Page: {meta.get('page_number', 'N/A')}"
            for meta in metadata
        ]
    )
    context = context.replace("'", "").replace('"', "").replace("\n", " ")
    return (
        f"Rules and Regulations for the Construction and Classification of Steel Ships. Answer the questions based on the context provided. "
        f"QUESTION: '{query}'\n\nCONTEXT: '{context}'\n\n"
        f"Metadata: {metadata_str}\n\nANSWER:"
    )


# Retrieve relevant context from vector database
def get_relevant_context_from_db(query, vector_db):
    try:
        search_results = vector_db.similarity_search(query, k=5)
        context = " ".join(result.page_content for result in search_results)
        metadata = [
            {
                "source": result.metadata.get("source", "N/A"),
                "page_number": result.metadata.get("page_number", "N/A"),
            }
            for result in search_results
        ]
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return "", []
    return context.strip(), metadata

# Generate an answer using Gemini 
def generate_answer(prompt):
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        answer = model.generate_content(prompt)
        return answer.text
    except Exception as e:
        return f"Error generating answer: {e}"

# Sidebar for collection selection and settings
with st.sidebar:
    st.title("Document Collection")
    collection_names = get_collections()
    selected_collection = st.selectbox("Select one", collection_names, key="collection")
    st.caption("Select a document collection to use for your queries.")

# Main Chat Interface
st.title("Enterprise Chatbot")
st.caption("Celerinn Technologies")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    if msg.get("metadata"):
        st.markdown("**Source Metadata:**")
        for meta in msg["metadata"]:
            source = meta.get("source", "N/A")
            page_number = meta.get("page_number", "N/A")
            st.caption(f"- **Source**: {source} | **Page**: {page_number}")


# Initialize vector database
vector_db = get_vector_db(selected_collection)

# Handle user input
if user_input := st.chat_input("Type your message here..."):
    # Add user's input to the conversation history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Retrieve context and metadata
    context, metadata = get_relevant_context_from_db(user_input, vector_db)
    if not context:
        response = "No relevant context found. Please try asking something else."
        response_metadata = []
    else:
        # Generate RAG prompt and get response
        prompt = generate_rag_prompt(query=user_input, context=context, metadata=metadata)
        response = generate_answer(prompt=prompt)
        response_metadata = metadata
    
    # Add the bot's response and metadata to the conversation history
    st.session_state.messages.append({"role": "assistant", "content": response, "metadata": response_metadata})
    st.chat_message("assistant").write(response)

    # Display metadata below the bot's response
    if response_metadata:
        st.markdown("**Source Metadata:**")
        for meta in response_metadata:
            source = meta.get("source", "N/A")
            page_number = meta.get("page_number", "N/A")
            st.caption(f"- **Source**: {source} | **Page**: {page_number}")


