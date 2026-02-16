import torch
from llm import Transformer
from tokenizer import Tokenizer


class Inference:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def response(self, query):
        ids = self.tokenizer.tokenize(query)
        ids_tensor = torch.tensor([ids])
        ids_in_context_window = ids_tensor[: -self.model.context_window]

        generated_response_token_ids = []
        generated_token_id = None
        end_token_id = tokenizer.tokenize("<END>")[0]

        while generated_token_id != end_token_id:
            logits = self.model(ids_tensor)
            guesses = self.model.decode(logits)

            new_token_id = guesses[0, -1]

            generated_response_token_ids.append(new_token_id)
            generated_token_id = new_token_id
            ids_in_context_window = torch.cat(
                [ids_in_context_window, torch.tensor([[new_token_id]])]
            )

            if ids_in_context_window.shape > self.model.context_window:
                ids_in_context_window = ids_in_context_window[1:]

        response = tokenizer.detokenize(generated_response_token_ids)
        return response

    def get_next_token(self, context_window):
        pass

    def format(self, response):
        return response


if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer.tokenize("What's up?")
    model = Transformer()
    inference = Inference()
