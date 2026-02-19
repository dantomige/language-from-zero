from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.block import TransformerBlock
from src.model.embeddings import EmbeddingLayer
from src.config.model import ModelConfig
import json
from pathlib import Path


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_blocks = config.n_blocks
        self.n_heads = config.n_heads
        self.context_window = config.context_window

        self.embedding_layer = EmbeddingLayer(
            self.vocab_size, self.d_model, self.context_window
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.d_model, self.n_heads) for _ in range(self.n_blocks)]
        )
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.linear = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of transformer model. Takes a dataset
        and predicts next word for each token.

        Args:
            X (torch.Tensor): dataset of size (batch_size, sequence_length)

        Returns:
            torch.Tensor: logits for each item token (batch_size, sequence_length, vocab_size)
        """
        embeddings = self.embedding_layer(X)

        for block in self.blocks:
            embeddings = block(embeddings)

        norm_embeddings = self.layer_norm(embeddings)

        logits = self.linear(norm_embeddings)

        return logits
    
    def save_pretrained(self, folder_path: str | Path, model_filename: str = "model_state.pt", config_filename: str = "config.json"):
        """Saves model state to a file.

        Args:
            folder_path (str): path to folder where model state file "model_state.pt" will be saved
        """
        path = Path(folder_path)
        torch.save(self.state_dict(), path / model_filename)
        self.config.save_pretrained(folder_path=folder_path, filename=config_filename)

    @classmethod
    def from_pretrained(
        cls,
        folder_path: str | Path,
        model_filename: str,
        config_filename: str,
    ) -> Transformer:
        """Loads model state from a file. Assumes model state and config are located in the given folder path.   

        Args:
            folder_path (str): path to folder containing model state file "model_state.pt"
        """
        folder_path = Path(folder_path)
        model_state_dict = torch.load(folder_path / model_filename)
        config = ModelConfig.from_pretrained(folder_path=folder_path, filename=config_filename)

        model = cls(config=config)
        model.load_state_dict(model_state_dict)
        return model

    def get_config(self):
        """Returns a copy ofthe model config."""
        return self.config.model_copy()

    def decode(self, logits):
        return torch.argmax(logits, dim=-1)

    def train(self):
        """Set model to train mode."""
        pass

    def eval(self):
        """Set model to eval mode"""
        pass


if __name__ == "__main__":
    vocab_size = 20
    d_model = 4
    n_blocks = 6
    context_window = 10

    X = torch.tensor([3, 4, 5, 3, 1, 2, 3, 10]).view(1, -1)

    print(X.shape)

    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_blocks=n_blocks,
        context_window=context_window,
    )

    print(transformer(X))
