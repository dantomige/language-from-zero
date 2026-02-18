import torch
from llm import Transformer
from tokenizer import Tokenizer


class Inference:

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def response(self, query, max_response_tokens, temperature = 1):
        ids = self.tokenizer.tokenize(query)
        ids_tensor = torch.tensor([ids])
        ids_in_context_window = ids_tensor[-self.model.context_window:]

        generated_response_token_ids = []

        for _iter in range(max_response_tokens):

            logits = self.model(ids_in_context_window) / temperature

            guesses = self.model.decode(logits)

            new_token_id = guesses[0, -1]

            generated_response_token_ids.append(new_token_id.item())
            
            ids_in_context_window = torch.cat(
                [ids_in_context_window, torch.tensor([[new_token_id]])]
            , dim=1)

            _, curr_window_size = ids_in_context_window.shape
            if  self.model.context_window < curr_window_size:
                ids_in_context_window = ids_in_context_window[:,1:]

        response = self.tokenizer.detokenize(generated_response_token_ids)
        return self.format(response)

    # def response(self, query, max_response_length = None):
    #     ids = self.tokenizer.tokenize(query)
    #     ids_tensor = torch.tensor([ids])
    #     ids_in_context_window = ids_tensor[: -self.model.context_window]

    #     generated_response_token_ids = []
    #     generated_token_id = None
    #     end_token_id = self.tokenizer.get_token("<END>")

    #     max_response_length = float("inf")

    #     while generated_token_id != end_token_id and max_response_length:
    #         logits = self.model(ids_tensor)
    #         guesses = self.model.decode(logits)

    #         new_token_id = guesses[0, -1]
    #         print(new_token_id)

    #         generated_response_token_ids.append(new_token_id)
    #         generated_token_id = new_token_id

    #         print(ids_in_context_window.shape)
    #         ids_in_context_window = torch.cat(
    #             [ids_in_context_window, torch.tensor([[new_token_id]])]
    #         )

    #         if ids_in_context_window.shape > self.model.context_window:
    #             ids_in_context_window = ids_in_context_window[1:]

    #         max_response_length -= 1

    #     response = self.tokenizer.detokenize(generated_response_token_ids)
    #     return self.format(response)

    def generate_next_token(self, ids_tensor):
        pass

    def format(self, response):
        return " ".join(response)


if __name__ == "__main__":
    # tokenizer = Tokenizer()
    # tokenizer.tokenize("What's up?")
    # model = Transformer()
    # inference = Inference()
    pass
