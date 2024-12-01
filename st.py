import streamlit as st
from main import preprocess_pdf, load_preprocessed_data, initialize_faiss, find_relevant_chunk, generate_answer_with_gemini, generate_answer_with_gemini_KEYWORD
from sentence_transformers import SentenceTransformer
import faiss

# Streamlit app title
st.title("AI Chatbot")

# Set paths and constants
PDF_PATH = "input.pdf"
CHUNK_FILE = "chunks.json"
EMBEDDING_FILE = "embeddings.npy"
SYSTEM_PROMPT = (
    "You are an AI chat bot to answer user questions on Student Resource Book of the NMIMS Global Access School For Continuing Education. "
    "I have provided you with the relevant chunks of information to answer the user's question under the tag {'CONTEXT'}. "
    "Following the CONTEXT, I have provided you with the user's question under the tag {'USER'S_QUESTION'}. "
    "Make sure the user's question is comprehensible and makes sense. If you are unsure what the user is asking for, reply requesting more information. "
    "Answer the user's question using only the context provided and do not use any other information. Do not hallucinate. "
    "Also if the user's question contain a greeting or any message for asking some assitance reply generously "
    "If the CONTEXT provided does not contain the information relevant to answer the user's question, reply with this STATEMENT: 'Sorry, I didn't understand your question. Do you want to connect with a live agent?'."
)
SYSTEM_PROMPT_KEYWORD = (
    "You are an AI Agent to help retrieve information from Student Resource Book of the NMIMS Global Access School For Continuing Education "
    "to answer user questions on that database. Extract the relevant keywords from the user query based on which the information will be extracted from the database. "
    "Provide me with those keywords."
)

# Initialize session state variables
if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "index" not in st.session_state:
    st.session_state.index = None

if "model" not in st.session_state:
    st.session_state.model = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Preprocess PDF and load data if not already done
if st.session_state.chunks is None or st.session_state.embeddings is None:
        preprocess_pdf(PDF_PATH, CHUNK_FILE, EMBEDDING_FILE)
        st.session_state.chunks, st.session_state.embeddings = load_preprocessed_data(CHUNK_FILE, EMBEDDING_FILE)

# Initialize FAISS index
if st.session_state.index is None:
        dimension = st.session_state.embeddings.shape[1]
        st.session_state.index = faiss.IndexFlatL2(dimension)
        st.session_state.index.add(st.session_state.embeddings)

# Load the SentenceTransformer model
if st.session_state.model is None:
        st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask your question here:"):
    # Display the user query in the chat
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Extract keywords from user query
    
    keywords = generate_answer_with_gemini_KEYWORD(SYSTEM_PROMPT_KEYWORD, prompt)

    # Find the most relevant chunk
    relevant_chunk = find_relevant_chunk(
            keywords,
            st.session_state.index,
            st.session_state.chunks,
            st.session_state.model
    )

    # Generate the assistant's response
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_text = "..."

        with st.spinner("Generating response..."):
            response_text = generate_answer_with_gemini(SYSTEM_PROMPT, prompt, relevant_chunk)

        # Display the response
        response_container.markdown(response_text)

    # Save the assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})