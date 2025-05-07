import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq  

client = Groq()

def load_vectorized_data(path):
    with open(path, "r") as f:
        return json.load(f)

def calculate_winning_percentage(team_name, vectorized_data, k=10):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = [entry["embedding"] for entry in vectorized_data]
    labels = [entry["label"] for entry in vectorized_data]

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    query_vec = model.encode([team_name], convert_to_numpy=True)
    _, indices = index.search(query_vec, k)

    matching_labels = [labels[i] for i in indices[0]]
    total = len(matching_labels)
    wins = matching_labels.count(team_name)
    percentage = (wins / total) * 100 if total > 0 else 0

    return wins, total, percentage

def stream_llm_response(team_name, wins, total, percentage):
    prompt = (
        f"The team '{team_name}' has played {total} matches and won {wins}. "
        f"Calculate and summarize their winning percentage in natural language."
    )

    print("\n=== Response ===\n")
    completion = client.chat.completions.create(
        model="llama3-70b-8192", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")

if __name__ == "__main__":

    vectorized_data = load_vectorized_data("data/JSON files/matches_vectorized.json")

    team_name = input("Enter the team name to calculate winning percentage: ")


    wins, total, percentage = calculate_winning_percentage(team_name, vectorized_data)
    stream_llm_response(team_name, wins, total, percentage)


#key: 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
#llama-3.3-70b-versatile