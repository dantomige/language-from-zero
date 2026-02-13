import torch
import torch.utils.data
from datasets import load_dataset

class LangaugeModelDataset(torch.utils.data):

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
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n".join(dataset['text'])

    print(type(full_text))