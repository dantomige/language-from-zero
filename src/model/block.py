import torch
import torch.nn as nn

from src.model.attention import CausalMultiHeadSelfAttention


class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.first_norm_layer = nn.LayerNorm(d_model)
        self.attention_layer = CausalMultiHeadSelfAttention(d_model, n_heads)
        self.second_norm_layer = nn.LayerNorm(d_model)
        self.FFNN_layer1 = nn.Linear(in_features=d_model, out_features=d_model * 4)
        self.FFNN_layer2 = nn.Linear(in_features=d_model * 4, out_features=d_model)

    def forward(self, x):

        norm_x = self.first_norm_layer(x)
        weighted_values = self.attention_layer(norm_x)
        norm_values = self.second_norm_layer(weighted_values + x)  # residual connection
        ffnn_out = self.FFNN_layer2(self.FFNN_layer1(norm_values))

        return ffnn_out + norm_values  # residual connection


if __name__ == "__main__":
    num_seq = 2
    d_model = 4
    x = torch.tensor([2, 3, 1, 2, 3, 4, 1, 6]).view(num_seq, d_model).float()
    X = x.unsqueeze(0)

    block = TransformerBlock(d_model)

    out = block(X)

    print(out)

    # print(x.shape, X.shape)

    # layer_norm = nn.LayerNorm(d_model)
    # batch_norm = nn.BatchNorm1d(d_model)

    # outx = layer_norm(x)
    # outX = batch_norm(X.transpose(1,2))

    # print(outx, outX.transpose(1,2))
