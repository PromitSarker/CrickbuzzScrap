import pandas as pd
import json
from sentence_transformers import SentenceTransformer

# Load the embedding model (can be swapped with another)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, ~384 dim

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")

def format_match_prompt(row):
    return (
        f"{row['team1']} played against {row['team2']} at {row['city']}. "
        f"The toss winner was {row['toss_winner']} and chose to {row['toss_decision']}. "
        f"{row['winner']} won the match."
    )

def format_points_prompt(row):
    return (
        f"Team {row['Team']} has played {row['Matches']} matches, with {row['Won']} wins "
        f"and {row['Lost']} losses. Points: {row['Points']}, NRR: {row['Net Run Rate']}."
    )

def vectorize_data(data, formatter, label_key=None):
    prompts = [formatter(row) for row in data]
    embeddings = model.encode(prompts, convert_to_numpy=True).tolist()
    
    result = []
    for i, row in enumerate(data):
        entry = {
            "input": prompts[i],
            "embedding": embeddings[i]
        }
        if label_key:
            entry["label"] = row[label_key]
        result.append(entry)
    return result

if __name__ == "__main__":
    # Matches
    matches = load_json("data/JSON files/matches_filtered.json")
    matches_vectors = vectorize_data(matches, format_match_prompt, label_key="winner")
    save_json(matches_vectors, "data/JSON files/matches_vectorized.json")

    # Points Table
    points = load_json("data/JSON files/points_table.json")
    points_vectors = vectorize_data(points, format_points_prompt, label_key="Team")
    save_json(points_vectors, "data/JSON files/points_table_vectorized.json")
