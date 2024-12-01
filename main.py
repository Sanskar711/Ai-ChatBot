import os
import json
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def split_text_into_chunks(text, max_length=200):
    """
    Split text into smaller chunks to process efficiently.
    """
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) <= max_length:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def preprocess_pdf(pdf_path, chunk_file, embedding_file, model_name="all-MiniLM-L6-v2"):
    """
    Extract text, create chunks, and save chunk embeddings to files.
    """
    # Check if preprocessing files already exist
    if os.path.exists(chunk_file) and os.path.exists(embedding_file):
        print("Preprocessed files already exist. Skipping preprocessing.")
        return

    # Step 1: Extract text
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Split into chunks
    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(text)

    # Step 3: Generate embeddings
    print("Generating embeddings for chunks...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    print(len(chunks))
    # Save chunks and embeddings
    print("Saving chunks and embeddings to files...")
    with open(chunk_file, "w") as f:
        json.dump(chunks, f)

    np.save(embedding_file, embeddings)


def load_preprocessed_data(chunk_file, embedding_file):
    """
    Load preprocessed chunks and embeddings from files.
    """
    print("Loading preprocessed chunks and embeddings...")
    with open(chunk_file, "r") as f:
        chunks = json.load(f)
    embeddings = np.load(embedding_file)
    return chunks, embeddings

import faiss
import requests

def initialize_faiss(embeddings):
    """
    Initialize a FAISS index with precomputed embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def find_relevant_chunk(query, index, chunks, model, top_k=15):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer_with_gemini_KEYWORD(system_prompt, query):
    # print('Context: ', context)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json",
        # "Authorization": f"Bearer {api_key}"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"SYSTEM_PROMPT: {system_prompt}\n\n USER'S_QUESTION: {query}\n\n"
                    }
                ]
            }
        ],
         "generation_config": {
            "temperature": 0.5
        }
    }
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json().get('candidates')[0].get('content').get('parts')[0].get('text',"Sorry, I didn’t understand your question. Do you want to connect with a live agent?")
    else:
        return f"Error: {response.status_code} - {response.text}"

def generate_answer_with_gemini(system_prompt, query, context):
    # print('Context: ', context)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json",
        # "Authorization": f"Bearer {api_key}"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"SYSTEM_PROMPT: {system_prompt}\n\n CONTEXT: {context}\n\n USER'S_QUERY: {query}\n\n ANSWER:"
                    }
                ]
            }
        ],
         "generation_config": {
            "temperature": 0.0
        }
    }
    response = requests.post(url, headers=headers, json=data)
    # print(response.status_code)
    # return response.json().get('candidates')[0].get('content').get('parts')[0].get('text')
    if response.status_code == 200:
        return response.json().get('candidates')[0].get('content').get('parts')[0].get('text')
    else:
        return f"Error: {response.status_code} - {response.text}"

# # Example usage
# api_key = "AIzaSyC2Sg8YRk_iOGWNg_sah-Teuylox9nanOY"
# # failure_statement = "Sorry, I didn't understand your question. Do you want to connect with a live agent?"
# System_Prompt = f"You are an AI chat bot to answer user questions on Student Resource Book of the NMIMS Global Access School For Continuing Education. I have provided you with the relevent chunks of information to answer the user's question under the tag {"CONTEXT"}. Following the CONTEXT I have provided you with the user's question inder the tag {"USER'S_QUESTION"}. Make sure the user's question is comprehendable and make's sense, if you are unsure of what the user is asking for, reply requesting for more information by the user. Answer the user's question using the only the context provided and do not use any other information. Do not Hallucinate. If the CONTEXT provided does not contain the information relevant to answer the user's question, reply with this STATEMENT: {"Sorry, I didn't understand your question. Do you want to connect with a live agent?"}."
# System_Prompt_KEYWORD = f"You are an AI Agent to help retrieve information from Student Resource Book of the NMIMS Global Access School For Continuing Education to answer user questions on that database. Extract the relevant Keywords from the user query on which according to whic the information will be extracted from the database. Provide me with those Keywords."
# user_query = input()
# # relevant_chunk = find_relevant_chunk(user_query, index, chunks, model)
# keywords = generate_answer_with_gemini_KEYWORD(System_Prompt_KEYWORD, user_query, api_key)
# print(keywords)
# relevant_chunk = find_relevant_chunk(keywords, index, chunks, model)

# answer = generate_answer_with_gemini(System_Prompt, user_query, relevant_chunk, api_key)
# print("Answer:", answer)