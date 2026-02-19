from pydantic import BaseModel

class ModelConfig(BaseModel):
    vocab_size: int
    d_model: int
    n_blocks: int
    num_heads: int
    context_window: int