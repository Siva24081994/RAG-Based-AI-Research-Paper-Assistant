import os
import faiss
import fitz  # PyMuPDF for text extraction
import torch
import gradio as gr
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pickle

# Load Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Hugging Face Model for Text Generation
qa_model = pipeline("text2text-generation", model="google/flan-t5-large")

# FAISS Index Settings
dimension = 384  
faiss_index = faiss.IndexFlatL2(dimension)

# Store Metadata (Chunk Text)
metadata_store = []

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def clean_text(text):
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,;?!\s]', '', text)
    return text.strip()

def chunk_text(text, chunk_size=1000):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def store_in_faiss(chunks):
    global faiss_index, metadata_store
    vectors = embedding_model.encode(chunks)
    vectors = np.array(vectors).astype('float32')
    faiss_index.add(vectors)
    metadata_store.extend(chunks)
    return len(chunks)

def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [metadata_store[i] for i in indices[0] if i < len(metadata_store)]

def generate_answer(query, retrieved_chunks):
    context = "\n".join(clean_text(chunk) for chunk in retrieved_chunks)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = qa_model(prompt, max_length=512, temperature=0.7, top_p=0.9)
    return response[0]['generated_text']

def process_pdf_and_query(pdf, query):
    text = extract_text_from_pdf(pdf)
    cleaned_text = clean_text(text)
    chunks = chunk_text(cleaned_text)
    store_in_faiss(chunks)
    retrieved_chunks = retrieve_relevant_chunks(query)
    return generate_answer(query, retrieved_chunks)

def gradio_interface(pdf, query):
    return process_pdf_and_query(pdf.name, query)

iface = gr.Interface(fn=gradio_interface, inputs=["file", "text"], outputs="text", title="DeepDive AI: Research Paper Insights")
iface.launch()
