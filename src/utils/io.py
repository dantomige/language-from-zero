import json
from pathlib import Path

def save_json(data: dict, filepath: str | Path) -> None:
    """Saves a dictionary to a json file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def read_json(filepath: str | Path) -> dict:
    """Reads a json file and returns a dictionary."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data