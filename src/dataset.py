import torch
from torch.utils.data import Dataset

class LangaugeModelDataset(Dataset):

    def __init__(self, tokens, context_window):
        self.tokens = tokens
        self.context_window = context_window

    def __len__(self):
        return len(self.tokens) - self.context_window - 1

    def __getitem__(self, index):
        chunk = self.tokens[index: index + self.context_window + 1]

        X, y = chunk[:-1], chunk[1:]

        return X, y


if __name__ == "__main__":
    pass