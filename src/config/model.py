from __future__ import annotations
from pydantic import BaseModel
from pathlib import Path
from src.utils.io import save_json, read_json

class ModelConfig(BaseModel):
    vocab_size: int
    d_model: int
    n_blocks: int
    n_heads: int
    context_window: int

    def save_pretrained(self, folder_path: str | Path, filename: str):
        path = Path(folder_path) / filename
        save_json(self.model_dump(), path)

    @classmethod
    def from_pretrained(cls, folder_path: str | Path, filename: str) -> ModelConfig:
        path = Path(folder_path) / filename
        
        # Read the raw JSON
        config_state = read_json(path)
            
        # Use Pydantic's internal validation to create the object
        return cls.model_validate(config_state)