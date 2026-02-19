import json

def save_json(data: dict, filepath: str):
    """Saves a dictionary to a json file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def read_json(filepath: str) -> dict:
    """Reads a json file and returns a dictionary."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data