import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, bias: bool = False):
        super().__init__()

        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, d_model = X.shape

        # print(X.shape)

        Q, K, V = self.W_Q(X), self.W_K(X), self.W_V(X)

        self_attention = Q @ K.transpose(-2, -1) / (d_model**0.5)

        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=X.device), diagonal=1
        ).bool()
        mask = mask.to(X.device)

        # print(mask)

        masked_self_attention = self_attention.masked_fill(mask == True, -float("inf"))

        # print(masked_self_attention)

        weighted_attention = F.softmax(masked_self_attention, dim=-1)

        # print(weighted_attention)

        weighted_values = weighted_attention @ V

        # print(weighted_values)

        return weighted_values


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, bias: bool = False):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"Cannot make a multi head attention layer with d_model: {d_model} and num_heads: {num_heads} (d_model % num_heads must equal 0)"
            )

        self.d_model = d_model
        self.num_heads = num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)

        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        nn.ModuleList

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, d_model = X.shape

        # get query, key, value
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # multi head matrices
        d_head = int(self.d_model / self.num_heads)

        Q_multihead = Q.view(batch_size, num_tokens, self.num_heads, d_head)
        K_multihead = K.view(batch_size, num_tokens, self.num_heads, d_head)
        V_multihead = V.view(batch_size, num_tokens, self.num_heads, d_head)

        # transpose for parallel head matrices
        Q_multihead = Q_multihead.transpose(1, 2)
        K_multihead = K_multihead.transpose(1, 2)
        V_multihead = V_multihead.transpose(1, 2)

        # masked attention
        multihead_self_attention = (
            Q_multihead @ K_multihead.transpose(-2, -1) / (d_head**0.5)
        )
        mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()
        mask = mask.to(X.device)
        masked_multihead_self_attention = multihead_self_attention.masked_fill(
            mask == True, -float("inf")
        )
        multihead_weighted_attention = F.softmax(
            masked_multihead_self_attention, dim=-1
        )

        # weighted values
        multihead_weighted_values = multihead_weighted_attention @ V_multihead

        # concat
        weighted_values_concat = (
            multihead_weighted_values.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, d_model)
        )

        # linear pass
        return self.W_O(weighted_values_concat)


if __name__ == "__main__":
    d_model = 8
    bias = False
    attention_layer = CausalSelfAttention(d_model, bias)
    X = torch.rand(1, 4, d_model)

    print(X)

    attention_layer.forward(X)
