import torch
import nltk
from nltk.corpus import gutenberg
from datasets import load_dataset
from src.tokenizer import Tokenizer
from src.data.dataset import LangaugeModelDataset
from src.services.content_filter import ContentFilterService

class DataProcessor:
    def __init__(self, tokenizer: Tokenizer, context_window: int):
        self.tokenizer = tokenizer
        self.context_window = context_window

    def from_nltk_gutenberg(self, num_books: int = None, update_vocab: bool = False) -> LangaugeModelDataset:
        """Creates a dataset to nltk gutenberg

        Args:
            limit (int, optional): max number of books to add. Defaults to None.
            update_vocab (bool, optional): whether to update the vocabulary of the tokenizer. Defaults to False.. Defaults to False.

        Returns:
            LangaugeModelDataset: dataset object containing tokenized text from nltk gutenberg
        """
        
        nltk.download("gutenberg")

        token_ids = []

        num_books_added = 0

        for book in gutenberg.fileids():

            if num_books is not None and num_books_added == num_books:
                break

            token_ids.append(self.tokenizer.get_id(self.tokenizer.START_TOKEN))
            raw_text = gutenberg.raw(book)
            token_ids.extend(self.tokenizer.tokenize(raw_text, update_vocab=update_vocab))
            token_ids.append(self.tokenizer.get_id(self.tokenizer.END_TOKEN))
            num_books_added += 1

        full_token_ids_tensor = torch.tensor(token_ids)

        dataset = LangaugeModelDataset(tokens=full_token_ids_tensor, context_window=self.context_window)
        return dataset

    def from_shareGPT(self, limit=None, update_vocab: bool = False) -> LangaugeModelDataset:
        """Creates a dataset from ShareGPT

        Args:
            limit (_type_, optional): the max number of conversations to process. Defaults to None.
            update_vocab (bool, optional): whether to update the vocabulary of the tokenizer. Defaults to False.

        Returns:
            LangaugeModelDataset: dataset object containing tokenized conversations from ShareGPT
        """
        
        dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    
        token_ids = []

        content_filter = ContentFilterService()

        num_conversations_added = 0

        for datapoint in dataset:
            if limit is not None and num_conversations_added == limit:
                break

            if datapoint['language'] != 'English':
                continue


            does_conversation_contain_code = any(content_filter.is_code(message['content']) for message in datapoint['conversation'])
            if does_conversation_contain_code:
                continue

            token_ids.append(self.tokenizer.get_id(self.tokenizer.START_TOKEN))

            for message in datapoint['conversation']:
                token_ids.append(self.tokenizer.get_id(self.tokenizer.HEADER_START_TOKEN))
                token_ids.append(self.tokenizer.get_id(message['role']))
                token_ids.append(self.tokenizer.get_id(self.tokenizer.HEADER_END_TOKEN))
                token_ids.extend(self.tokenizer.tokenize(message['content'], update_vocab=update_vocab))
                token_ids.append(self.tokenizer.get_id(self.tokenizer.END_OF_TURN_TOKEN))

            token_ids.append(self.tokenizer.get_id(self.tokenizer.END_TOKEN))

            num_conversations_added += 1

        full_token_ids_tensor = torch.tensor(token_ids)
            
        return LangaugeModelDataset(tokens=full_token_ids_tensor, context_window=self.context_window)
    
    def from_diary_of_a_ceo_podcast(self, update_vocab: bool = False) -> LangaugeModelDataset:
        raise NotImplementedError("This method is not implemented yet")
    