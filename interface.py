import json
import torch
import gradio as gr
from tokenizer import Tokenizer
from llm import Transformer
from inference import Inference


def load_tokenizer(checkpoint):

    tokenizer_file = checkpoint["tokenizer_file"]
    with open(tokenizer_file, "r") as f:
        data = json.load(f)

    token_to_id = data["token_to_id"]
    tokenizer = Tokenizer.update(token_to_id=token_to_id)
    return tokenizer


def load_llm_model(checkpoint):

    vocab_size, d_model, n_blocks, context_window = (
        checkpoint["vocab_size"],
        checkpoint["d_model"],
        checkpoint["n_blocks"],
        checkpoint["context_window"],
    )
    llm_model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_blocks=n_blocks,
        max_seq_len=context_window,
    )
    return llm_model


def load_inference_model(filepath):

    checkpoint = torch.load(filepath)

    tokenizer = load_tokenizer(checkpoint=checkpoint)
    llm_model = load_llm_model(checkpoint=checkpoint)

    inference_model = Inference(llm_model, tokenizer)

    return inference_model


class ModelInterface:

    def __init__(self, inference_model):
        self.inference_model = inference_model

    def predict(self, query):
        return self.inference_model.response(query)


def process(user_input):
    return f"Model processed: {user_input}"


def main():
    FILEPATH = ""
    inference_model = load_inference_model(FILEPATH)
    model_interface = ModelInterface(inference_model=inference_model)

    interface = gr.Interface(fn=model_interface.predict, inputs="text", outputs="text")
    interface.launch()


if __name__ == "__main__":
    main()
