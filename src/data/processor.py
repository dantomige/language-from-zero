import torch
import nltk
from nltk.corpus import gutenberg
from datasets import load_dataset
from src.tokenizer import Tokenizer
from src.data.dataset import LangaugeModelDataset

class DataProcessor:
    def __init__(self, tokenizer: Tokenizer, context_window: int):
        self.tokenizer = tokenizer
        self.context_window = context_window

    def from_nltk_gutenberg(self) -> LangaugeModelDataset:
        nltk.download("gutenberg")

        token_ids = []

        for book in gutenberg.fileids():
            token_ids.append(self.tokenizer.get_id(self.tokenizer.START_TOKEN))
            raw_text = gutenberg.raw(book)
            token_ids.extend(self.tokenizer.tokenize(raw_text))
            token_ids.append(self.tokenizer.get_id(self.tokenizer.END_TOKEN))

        full_token_ids_tensor = torch.tensor(token_ids)

        dataset = LangaugeModelDataset(tokens=full_token_ids_tensor, context_window=self.context_window)
        return dataset

    def from_shareGPT(self) -> LangaugeModelDataset:
        raise NotImplementedError("This method is not implemented yet")
    
    def from_diary_of_a_ceo_podcast(self):
        raise NotImplementedError("This method is not implemented yet")
    