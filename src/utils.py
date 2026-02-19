import json


def save_as_json(data, filename: str) -> None:
    """Saves data to filename as json

    Args:
        data (dict): represents data to store
        filename (str): name of location to store data in current working directory

    Returns:
        None
    """
    with open(filename, "w") as f:
        json.dump(data, f)
