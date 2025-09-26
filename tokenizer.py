import re

class Tokenizer:

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.index = 0

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
    
    def detokenize(self, ids):
        return [self.id_to_token[id] for id in ids]
    
    def num_tokens(self):
        return self.index


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
    other_tkns =tknizr.tokenize(other_text)

    print(tknizr.id_to_token.values())