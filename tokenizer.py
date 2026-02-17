import re
import torch
import torch.nn.functional as F

class Tokenizer:
    
    DEFAULT_SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    UNKNOWN_TOKEN = "<UNK>"

    def __init__(self, special_tokens = None):
        self.vocab_size = 0
        self.token_to_id = {}
        self.id_to_token = {}

        tokens_to_add = special_tokens if special_tokens is not None else self.DEFAULT_SPECIAL_TOKENS
        self.special_tokens = tokens_to_add


        for word in tokens_to_add:
            self.add_to_vocab(word)

    def tokenize(self, text, update_vocab=False):
        """Takes raw text and produces numerical encoding of its tokens.

        Args:
            text (str): complete raw text
            update_vocab (bool, optional): bool to update unknown words or use unknown token. Defaults to False.

        Returns:
            List[int]: ids for each token in text
        """

        tokens = self.segment(text)

        ids = []
        for token in tokens:
            if token not in self.token_to_id:
                if not update_vocab:
                    token = self.UNKNOWN_TOKEN
                else:
                    self.add_to_vocab(token)
                    
            ids.append(self.token_to_id[token])

        return ids
    
    def segment(self, text: str):
        text = text.lower()
        tokens = re.findall(r'\d+|\w+|[^\w\s]', text)
        return tokens
    
    def get_id(self, token: str):
        token = token.lower()
        return self.token_to_id.get(token, None)

    def get_token(self, id):
        return self.id_to_token.get(id, None)
    
    def add_to_vocab(self, word: str):
        """Adds word to the vocab. Only changes vocabulary if word is not already in the vocabulary

        Args:
            word (str): word to add to vocab
        """
        if word not in self.special_tokens:
            word = word.lower()
        
        if word in self.token_to_id:
            return
        
        word_id = self.vocab_size
        self.token_to_id[word] = word_id
        self.id_to_token[word_id] = word
        self.vocab_size += 1


    def detokenize(self, ids):
        tokens = []
        for id in ids:
            token = self.get_token(id)
            if token is None:
                raise ValueError(f"All ids must be present in the tokenizer. {id} not present in tokenizer")
            tokens.append(token)
        return tokens

    def load_from_state_dict(self, state_dict):
        self.vocab_size, self.token_to_id, self.id_to_token = state_dict["vocab_size"], state_dict["token_to_id"], state_dict["id_to_token"]

    def state_dict(self):
        state_dict = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token
        }
        return state_dict


if __name__ == "__main__":
    tknizr = Tokenizer()

    text = '''
    On March 14, 2025, Dr. Eleanor Finch stood at the podium and declared: “We’ve reached 97.3% accuracy on the model—an unprecedented milestone!” 
    The audience erupted, clapping, cheering, even whispering: “Is this real? Can it scale?”  

    Meanwhile, outside the conference hall, three students argued:  
    1) “AI will save the world.”  
    2) “AI will destroy the world.”  
    3) “AI is just another tool—like the printing press.”  

    Eleanor sighed. She remembered her grandfather saying, “Progress is neither good nor bad, it simply *is*.”  

    Emails buzzed in her inbox:  
    - from-team@example.com  
    - urgent@investors.net  
    - spam_offer@random.biz  

    She scrolled quickly, ignoring most of them, until one caught her eye: *CONFIDENTIAL: Prototype Release v2.0 Scheduled.*  

    Later that night—around 11:45 p.m.—she opened her journal and wrote:  
    “Today I stood before 2,000 people. Tomorrow, I’ll return to the lab, where it’s just me, the hum of GPUs, and a blinking cursor.  
    Funny how silence feels louder than applause.”  

    Then she closed the book, turned off the lamp, and whispered: “Let’s see what tomorrow brings.”
    '''
    
    other_text = '''
    The train screeched into the station at exactly 6:42 a.m., steam hissing and wheels grinding against the rails. 
    Commuters shuffled forward, eyes glued to their phones, earbuds in, coffee cups half-empty. 
    Among them was Daniel, carrying a worn leather briefcase and a secret he had sworn never to tell. 
    The announcement echoed overhead: “Next stop, Riverside.” 
    He hesitated, staring at the sign, wondering if this was the day everything would change.
    '''

    tkns = tknizr.tokenize(text)
    other_tkns = tknizr.tokenize(other_text)

    # print(tknizr.id_to_token.values())
    
    vocab_size = tknizr.vocab_size()
    # print(vocab_size)
    seq_len = 10

    logits = torch.randint(0, 10, (1, seq_len, vocab_size)).float()

    # print(logits)

    tknizr.decode(logits)

