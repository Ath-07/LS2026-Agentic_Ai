import re
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================================================
# LOAD DOCUMENT
# =====================================================

with open("document.txt", "r", encoding="utf-8") as f:
    text = f.read()

# =====================================================
# CHUNKING STRATEGIES
# =====================================================

def fixed_chunk(text, size=300, overlap=50):

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        chunk = words[start:start + size]

        chunks.append(" ".join(chunk))

        start += (size - overlap)

    return chunks


def sentence_chunk(text):

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []

    i = 0

    while i < len(sentences):

        chunk = " ".join(sentences[i:i + 5])

        chunks.append(chunk)

        i += 4

    return chunks


def sliding_window_chunk(text, window=400, step=100):

    words = text.split()

    chunks = []

    for start in range(0, len(words), step):

        chunk = words[start:start + window]

        if len(chunk) == 0:
            break

        chunks.append(" ".join(chunk))

    return chunks


# =====================================================
# CHUNK STATISTICS
# =====================================================

def chunk_stats(chunks):

    lengths = [len(chunk.split()) for chunk in chunks]

    return {
        "num_chunks": len(chunks),
        "mean_len": np.mean(lengths),
        "min_len": np.min(lengths),
        "max_len": np.max(lengths)
    }


# =====================================================
# GENERATE CHUNKS
# =====================================================

fixed_chunks = fixed_chunk(text)

sentence_chunks = sentence_chunk(text)

sliding_chunks = sliding_window_chunk(text)

# =====================================================
# PRINT PART A RESULTS
# =====================================================

print("=" * 60)
print("PART A: CHUNKING STRATEGY COMPARISON")
print("=" * 60)

strategies = {
    "Fixed-size": fixed_chunks,
    "Sentence-based": sentence_chunks,
    "Sliding Window": sliding_chunks
}

for name, chunks in strategies.items():

    stats = chunk_stats(chunks)

    print(f"\n{name}")

    print(f"Number of Chunks : {stats['num_chunks']}")

    print(f"Mean Length      : {stats['mean_len']:.2f}")

    print(f"Min Length       : {stats['min_len']}")

    print(f"Max Length       : {stats['max_len']}")

# =====================================================
# EMBEDDING MODEL
# =====================================================

print("\n")
print("=" * 60)
print("LOADING EMBEDDING MODEL")
print("=" * 60)

model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# COSINE SIMILARITY
# =====================================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

# =====================================================
# RETRIEVER
# =====================================================

def build_embeddings(chunks):

    return model.encode(chunks)


def retrieve(query,
             chunks,
             embeddings,
             top_k=3):

    query_embedding = model.encode(query)

    scores = []

    for idx, emb in enumerate(embeddings):

        score = cosine_similarity(
            query_embedding,
            emb
        )

        scores.append((idx, score))

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in scores[:top_k]:

        results.append(
            (chunks[idx], score)
        )

    return results


# =====================================================
# MANUAL QA BENCHMARK
# =====================================================
#
# IMPORTANT:
# Replace answers if needed so they EXACTLY
# appear somewhere in your document.
#
# String matching is used.
#
# =====================================================

qa_pairs = [

    (
        "Why is leadership important?",
        "influence"
    ),

    (
        "Can managers be trained to become leaders?",
        "leadership can be trained"
    ),

    (
        "What skill helps managers gain support?",
        "social skills"
    ),

    (
        "What is one component of emotional intelligence?",
        "self-awareness"
    ),

    (
        "Why should managers understand others?",
        "empathy"
    )
]

# =====================================================
# HIT RATE EVALUATION
# =====================================================

def evaluate_strategy(chunks,
                      embeddings,
                      qa_pairs):

    hits = 0

    for question, answer in qa_pairs:

        retrieved_chunks = retrieve(
            question,
            chunks,
            embeddings,
            top_k=3
        )

        combined_text = " ".join(

            chunk.lower()

            for chunk, score

            in retrieved_chunks

        )

        if answer.lower() in combined_text:

            hits += 1

    return hits


# =====================================================
# BUILD EMBEDDINGS
# =====================================================

fixed_embeddings = build_embeddings(
    fixed_chunks
)

sentence_embeddings = build_embeddings(
    sentence_chunks
)

sliding_embeddings = build_embeddings(
    sliding_chunks
)

# =====================================================
# PART B RESULTS
# =====================================================

print("\n")
print("=" * 60)
print("PART B: RETRIEVAL COMPARISON")
print("=" * 60)

results = []

for name, chunks, embeddings in [

    ("Fixed-size",
     fixed_chunks,
     fixed_embeddings),

    ("Sentence-based",
     sentence_chunks,
     sentence_embeddings),

    ("Sliding Window",
     sliding_chunks,
     sliding_embeddings)

]:

    hits = evaluate_strategy(
        chunks,
        embeddings,
        qa_pairs
    )

    stats = chunk_stats(chunks)

    results.append(
        [
            name,
            stats["num_chunks"],
            round(stats["mean_len"], 2),
            f"{hits}/5"
        ]
    )

# =====================================================
# FINAL TABLE
# =====================================================

print()

print(
    f"{'Strategy':20}"
    f"{'Chunks':10}"
    f"{'Mean Len':12}"
    f"{'Hit Rate'}"
)

print("-" * 60)

for row in results:

    print(
        f"{row[0]:20}"
        f"{row[1]:10}"
        f"{row[2]:12}"
        f"{row[3]}"
    )

# =====================================================
# OPTIONAL:
# SHOW RETRIEVAL EXAMPLES
# =====================================================

print("\n")
print("=" * 60)
print("SAMPLE RETRIEVALS")
print("=" * 60)

sample_question = qa_pairs[0][0]

print("\nQuestion:")
print(sample_question)

retrieved = retrieve(
    sample_question,
    fixed_chunks,
    fixed_embeddings,
    top_k=3
)

for rank, (chunk, score) in enumerate(
        retrieved,
        start=1):

    print(f"\nRank {rank}")

    print(
        f"Score: {score:.4f}"
    )

    print(
        chunk[:250].replace(
            "\n",
            " "
        ) + "..."
    )