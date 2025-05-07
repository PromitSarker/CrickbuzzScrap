import pandas as pd
import json

def convert_matches_csv(csv_path, json_output_path):
    df = pd.read_csv(csv_path)
    
    remove = [
        "season", "umpire1", "umpire2", "reserve_umpire", 
        "match_referee", "reserve_umpire_city", "venue"
    ]

    df = df.drop(columns=remove, errors='ignore')

    df.to_json(json_output_path, orient="records", indent=2)
    print(f"Saved cleaned matches JSON to: {json_output_path}")

def convert_points_table_csv(csv_path, json_output_path):
    df = pd.read_csv(csv_path)
    df.to_json(json_output_path, orient="records", indent=2)
    print(f"Saved points table JSON to: {json_output_path}")

if __name__ == "__main__":
    convert_matches_csv("data/CSV files/matches.csv", "data/JSON files/matches_filtered.json")
    convert_points_table_csv("data/CSV files/points_table.csv", "data/JSON files/points_table.json")
