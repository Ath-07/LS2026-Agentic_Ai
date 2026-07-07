import matplotlib.pyplot as plt

# =====================================================
# QUESTION 3 - PART A
# DECISION TREE
# =====================================================

def recommend_approach(scenario):

    if (
        scenario["data_changes_frequently"]
        and scenario["knowledge_type"] == "behavioral"
    ):
        return (
            "RAG + Fine-Tuning",
            "Knowledge changes frequently while custom behavior is required."
        )

    elif scenario["data_changes_frequently"]:
        return (
            "RAG",
            "Frequently changing information is best handled through retrieval."
        )

    elif scenario["knowledge_type"] == "behavioral":
        return (
            "Fine-Tuning",
            "The task primarily requires learning a specific behavior."
        )

    elif (
        scenario["budget"] == "low"
        and not scenario["need_specific_output_format"]
    ):
        return (
            "Prompt Engineering only",
            "Simple prompting is sufficient for this task."
        )

    else:
        return (
            "RAG",
            "Retrieval offers the most flexible solution."
        )


# =====================================================
# TEST SCENARIOS
# =====================================================

scenarios = [

    {
        "name": "Company Knowledge Base",
        "data_changes_frequently": True,
        "need_specific_output_format": False,
        "budget": "medium",
        "latency_sensitive": False,
        "knowledge_type": "factual"
    },

    {
        "name": "Customer Support Bot",
        "data_changes_frequently": True,
        "need_specific_output_format": True,
        "budget": "high",
        "latency_sensitive": False,
        "knowledge_type": "behavioral"
    },

    {
        "name": "Email Rewriter",
        "data_changes_frequently": False,
        "need_specific_output_format": False,
        "budget": "low",
        "latency_sensitive": False,
        "knowledge_type": "factual"
    },

    {
        "name": "Medical Assistant",
        "data_changes_frequently": False,
        "need_specific_output_format": True,
        "budget": "high",
        "latency_sensitive": True,
        "knowledge_type": "behavioral"
    },

    {
        "name": "Research Assistant",
        "data_changes_frequently": True,
        "need_specific_output_format": False,
        "budget": "medium",
        "latency_sensitive": False,
        "knowledge_type": "factual"
    }

]

print("=" * 60)
print("QUESTION 3 - PART A")
print("DECISION TREE")
print("=" * 60)

for scenario in scenarios:

    recommendation, justification = recommend_approach(
        scenario
    )

    print("\nScenario:", scenario["name"])

    print("Recommendation:", recommendation)

    print("Justification:", justification)

# =====================================================
# QUESTION 3 - PART B
# HALLUCINATION STRESS TEST
# =====================================================

queries = [

    ("Why is leadership important?", True),

    ("Can managers be trained to become leaders?", True),

    ("What are components of emotional intelligence?", True),

    ("Who won FIFA World Cup 2022?", False),

    ("What is Bitcoin?", False),

    ("Who is the President of France?", False)

]

scores = []
labels = []
colors = []

print("\n")
print("=" * 60)
print("QUESTION 3 - PART B")
print("HALLUCINATION STRESS TEST")
print("=" * 60)

for query, answerable in queries:

    # Top chunk only
    results = retrieve(query, top_k=1)

    top_chunk, score = results[0]

    answer = rag_answer(query)

    print("\nQUERY:")
    print(query)

    print("\nTOP RETRIEVED CHUNK (FIRST 100 WORDS):")

    preview = " ".join(
        top_chunk.split()[:100]
    )

    print(preview)

    print("\nCOSINE SIMILARITY SCORE:")

    print(round(score, 4))

    print("\nLLM ANSWER:")

    print(answer)

    print("\n" + "-" * 60)

    scores.append(score)

    labels.append(
        query[:20]
    )

    colors.append(
        "green" if answerable else "red"
    )

# =====================================================
# BAR CHART
# =====================================================

plt.figure(figsize=(10, 5))

plt.bar(
    labels,
    scores,
    color=colors
)

plt.title(
    "Cosine Similarity Scores for Queries"
)

plt.xlabel(
    "Queries"
)

plt.ylabel(
    "Cosine Similarity"
)

plt.xticks(
    rotation=25
)

plt.tight_layout()

plt.savefig(
    "question3_similarity_plot.png"
)

print("\nBar chart saved as:")
print("question3_similarity_plot.png")

plt.show()