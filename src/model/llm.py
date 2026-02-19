import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.block import TransformerBlock
from src.model.embeddings import EmbeddingLayer
from src.config.model import ModelConfig
import json
from pathlib import Path
from typing import Union


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_blocks = config.n_blocks
        self.n_heads = config.num_heads
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
