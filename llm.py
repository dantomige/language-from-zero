import torch
import torch.nn as nn
import torch.nn.functional as F
from block import TransformerBlock
from embeddings import EmbeddingLayer

class Transformer(nn.Module):
    
    def __init__(self, vocab_size, d_model, n_blocks, max_seq_len):
        super().__init__()

        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_len)
        self.blocks = [TransformerBlock(d_model) for _ in range(n_blocks)]
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)


    def forward(self, X):
        embeddings = self.embedding_layer(X)

        for block in self.blocks:
            embeddings = block(embeddings)

        norm_embeddings = self.layer_norm(embeddings)

        logits = self.linear(norm_embeddings)

        return logits
    
    def decode(self, logits):
        return torch.argmax(logits, dim=-1)
    

if __name__ == "__main__":
    vocab_size = 20
    d_model = 4
    n_blocks = 6
    max_seq_len = 10

    X = torch.tensor([3, 4, 5, 3, 1, 2, 3, 10]).view(1,-1)
    
    print(X.shape)
    
    transformer = Transformer(vocab_size=vocab_size, d_model=d_model, n_blocks=n_blocks, max_seq_len=max_seq_len)

    print(transformer(X))