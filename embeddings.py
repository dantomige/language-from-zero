import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    
    def __init__(self, vocab_size, d_model, max_seq_length):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.positional_embeddings = self._generate_positional_embeddings(max_seq_length, d_model)

    def _generate_positional_embeddings(self, max_seq_length, d_model):
        positional_embeddings = torch.zeros(max_seq_length, d_model)
        positions = torch.arange(0, max_seq_length).unsqueeze(1)
        trig_arguments_column_factors = 10000 ** (torch.arange(0, d_model, 2)/d_model)
        # print(trig_arguments_column_factors)
        even_positional_embedding_values = torch.sin(positions / trig_arguments_column_factors)
        odd_positional_embedding_values = torch.cos(positions / trig_arguments_column_factors)
        
        positional_embeddings[:, ::2] = even_positional_embedding_values
        positional_embeddings[:,1::2] = odd_positional_embedding_values

        # print(positional_embeddings)
        self.register_buffer("positional_embeddings", positional_embeddings)
        return positional_embeddings

    def forward(self, x):
        return self.word_embeddings(x) + self.positional_embeddings[:x.shape[1], :]


if __name__ == "__main__":

    vocab_size = 5
    d_model = 4
    max_seq_length = 10

    X = torch.tensor([0, 2, 4, 2, 1]).view(1, -1)

    # print(X.shape)
    # print(X)

    embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_length)

    embeddings = embedding_layer(X)

    # print(embeddings)