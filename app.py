import json
import torch
import gradio as gr
from tokenizer import Tokenizer
from src.model.llm import Transformer
from inference import Inference

def normalize(tokenizer_state_dict):
    _, _, id_to_tokens = tokenizer_state_dict["vocab_size"], tokenizer_state_dict["token_to_id"], tokenizer_state_dict["id_to_token"]
    id_to_tokens = {int(token_id): token for token_id, token in id_to_tokens.items()}
    tokenizer_state_dict["id_to_token"] = id_to_tokens
    return tokenizer_state_dict

def load_tokenizer(checkpoint):

    tokenizer_filename = checkpoint["tokenizer_filename"]
    with open(tokenizer_filename, "r") as f:
        tokenizer_state_dict = json.load(f)

    tokenizer_state_dict = normalize(tokenizer_state_dict)
    tokenizer = Tokenizer()
    tokenizer.load_from_state_dict(tokenizer_state_dict)

    return tokenizer


def load_llm_model(checkpoint):

    vocab_size, d_model, n_blocks, n_heads, context_window = (
        checkpoint["vocab_size"],
        checkpoint["d_model"],
        checkpoint["n_blocks"],
        checkpoint["n_heads"],
        checkpoint["context_window"],
    )

    model_state_dict = checkpoint["model_state_dict"]

    llm_model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=n_heads,
        max_seq_len=context_window,
    )
    llm_model.load_state_dict(model_state_dict)

    return llm_model


def load_inference_model(filepath):

    checkpoint = torch.load(filepath)

    tokenizer = load_tokenizer(checkpoint=checkpoint)
    llm_model = load_llm_model(checkpoint=checkpoint)

    inference_model = Inference(llm_model, tokenizer)

    return inference_model


class ModelInterface:

    def __init__(self, inference_model: Inference):
        self.inference_model = inference_model

    def predict(self, query, max_response_tokens = 50):
        return self.inference_model.response(query, max_response_tokens=max_response_tokens)


def main():
    FILENAME = "checkpoint_2026-02-18 18:54:19.158793+00:00.pth"

    inference_model = load_inference_model(FILENAME)
    # print(inference_model.tokenizer.token_to_id)

    model_interface = ModelInterface(inference_model=inference_model)

    interface = gr.Interface(fn=model_interface.predict, inputs="text", outputs="text")
    interface.launch()


if __name__ == "__main__":
    main()
