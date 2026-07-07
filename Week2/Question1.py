import os

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typer import prompt
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Log in by passing your token directly
login(token=os.getenv("HF_TOKEN"))

# =====================================================
# PART A: CHUNKING
# =====================================================

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40):
    """
    Split text into chunks with overlap.
    """
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += (chunk_size - overlap)

    return chunks


# =====================================================
# LOAD DOCUMENT
# =====================================================

with open("document.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_text(text)

print("=" * 60)
print("PART A: CHUNKING")
print("=" * 60)
print(f"Total Chunks Created: {len(chunks)}")
print()

# =====================================================
# PART B: EMBEDDINGS + RETRIEVAL
# =====================================================

print("=" * 60)
print("LOADING EMBEDDING MODEL...")
print("=" * 60)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all chunks
chunk_embeddings = model.encode(chunks)

print("Embedding Shape:", chunk_embeddings.shape)
print()


def cosine_similarity(a, b):
    """
    Manual cosine similarity implementation
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query: str, top_k: int = 3):
    """
    Retrieve top-k most similar chunks.
    Returns:
        [(chunk, score), ...]
    """

    query_embedding = model.encode(query)

    similarities = []

    for idx, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        similarities.append((idx, score))

    similarities.sort(key=lambda x: x[1], reverse=True)

    results = []

    for idx, score in similarities[:top_k]:
        results.append((chunks[idx], score))

    return results


# =====================================================
# TEST RETRIEVAL
# =====================================================

queries = [
    "What is the main topic of the document?",
    "What are the important concepts discussed?",
    "Summarize the key ideas."
]

print("=" * 60)
print("PART B: RETRIEVAL RESULTS")
print("=" * 60)

for query in queries:

    print(f"\nQUERY: {query}")
    print("-" * 60)

    results = retrieve(query, top_k=3)

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Cosine Score: {score:.4f}")

        preview = chunk[:300].replace("\n", " ")
        print(f"Chunk Preview:\n{preview}...")
        print()

# =====================================================
# PART C: GENERATION (FLAN-T5 BASE)
# =====================================================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("=" * 60)
print("LOADING FLAN-T5 BASE...")
print("=" * 60)

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def rag_answer(query):

    # Retrieve top 3 chunks
    results = retrieve(query, top_k=3)

    # Keep only first 80 words from each chunk
    # to avoid exceeding model context length
    shortened_chunks = []

    for chunk, score in results:
        shortened_chunk = " ".join(chunk.split()[:80])
        shortened_chunks.append(shortened_chunk)

    context = "\n\n".join(shortened_chunks)

    prompt = f"""
Answer the question using ONLY the context below.

If the answer is not present in the context,
reply exactly:
I don't know

Context:
{context}

Question:
{query}

Answer:
"""

    # Tokenize with truncation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = llm.generate(
        **inputs,
        max_new_tokens=100
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# =====================================================
# TEST RAG
# =====================================================

print("\n")
print("=" * 60)
print("PART C: RAG GENERATION")
print("=" * 60)

# Query from document
query1 = "What is the main topic of the document?"

answer1 = rag_answer(query1)

print("\nANSWERABLE QUERY")
print("-" * 60)
print("Question:", query1)
print("Answer:", answer1)


# Out of scope query
query2 = "Who won the FIFA World Cup 2022?"

answer2 = rag_answer(query2)

print("\nOUT-OF-SCOPE QUERY")
print("-" * 60)
print("Question:", query2)
print("Answer:", answer2)