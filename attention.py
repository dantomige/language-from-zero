import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, bias):
        super().__init__()

        self.d_model = d_model
    
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        batch, num_tokens, d_model = X.shape

        # print(X.shape)

        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)

        self_attention = Q @ K.transpose(-2, -1)

        mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()

        # print(mask)

        masked_self_attention = self_attention.masked_fill(mask == True, -float("inf"))

        # print(masked_self_attention)

        weighted_attention = F.softmax(masked_self_attention, dim=-1)

        # print(weighted_attention)

        weighted_values = weighted_attention @ V

        # print(weighted_values)

        return weighted_values

        

if  __name__ == "__main__":
    d_model = 8
    bias = False
    attention_layer = CausalSelfAttention(d_model, bias)
    X = torch.rand(1, 4, d_model)

    print(X)

    attention_layer.forward(X)