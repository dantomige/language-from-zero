import re
import json
import torch
import torch.nn.functional as F

class Tokenizer:

    def __init__(self, special_tokens=None):
        self.token_to_id = special_tokens if special_tokens is not None else {}
        self.id_to_token = {token_id: special_token for special_token, token_id in special_tokens.items()} if special_tokens is not None else {}
        self.index = len(special_tokens) if special_tokens is not None else 0

    def tokenize(self, text):
        tokens = re.findall(r'\d+|\w+|[^\w\s]', text)
        ids = []
        for token in tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.index
                self.id_to_token[self.index] = token
                self.index += 1
            
            ids.append(self.token_to_id[token])

        return ids
    
    def add_to_vocab(self, text):
        pass

    def update(self, token_to_id):
        pass
    
    def detokenize(self, ids):
        return [self.id_to_token[id] for id in ids]

    def vocab_size(self):
        return self.index
    
    def save_tokenizer(self, path):
        data = {
            "vocab_size": self.index,
            "token_to_id": self.token_to_id
        }

        with open(path, "w") as f:
            json.dump(data, f)


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

