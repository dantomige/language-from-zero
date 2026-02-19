from pathlib import Path
from pydantic import BaseModel

class ModelConfig(BaseModel):
    vocab_size: int
    d_model: int
    n_blocks: int
    num_heads: int
    context_window: int

    def save(self, filepath):
        """Saves the config to a JSON file."""
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load(cls, filepath):
        """Loads and validates a config from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No config found at {filepath}")
            
        json_str = path.read_text(encoding='utf-8')
        
        return cls.model_validate_json(json_str)