import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EncodingLayer(nn.Module):

    def __init__(self, vocab_size, dim_size):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim_size)
        self.positional_encoding = None

class AttentionBlock(nn.Module):

    def __init__(self):
        super().__init__()
        pass

class TransformerBlock(nn.Module):
    
    def __init__(self):
        super().__init__()
        pass


class LanguageModel(nn.Module):
    
    def __init__(self, vocab_size, dim_size):
        super().__init__()

        self.encoding_layer = EncodingLayer(vocab_size=vocab_size, dim_size=dim_size)

        self.block1 = TransformerBlock()
        self.block2 = TransformerBlock()
        self.block3 = TransformerBlock()
        self.block4 = TransformerBlock()
        self.block5 = TransformerBlock()
        self.block6 = TransformerBlock()

        self.norm = nn.LayerNorm()
        self.linear = nn.Linear()
        self.softmax = nn.Softmax()

    def forward(self, X):
        pass

    def backward(self):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def train(self, X, y):
        pass


if __name__ == "__main__":
    model = LanguageModel()
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
