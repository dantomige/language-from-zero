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

    tokenizer = Tokenizer()
    tokenizer.load_from_state_dict(tokenizer_state_dict)

    return tokenizer


def load_llm_model(checkpoint):

    vocab_size, d_model, n_blocks, context_window = (
        checkpoint["vocab_size"],
        checkpoint["d_model"],
        checkpoint["n_blocks"],
        checkpoint["context_window"],
    )

    model_state_dict = checkpoint["model_state_dict"]

    llm_model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_blocks=n_blocks,
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

    def __init__(self, inference_model):
        self.inference_model = inference_model

    def predict(self, query):
        return self.inference_model.response(query)


def process(user_input):
    return f"Model processed: {user_input}"


def main():
    FILEPATH = "checkpoint_2026-02-17 01:46:08.193650+00:00.pth"

    inference_model = load_inference_model(FILEPATH)

    model_interface = ModelInterface(inference_model=inference_model)

    interface = gr.Interface(fn=model_interface.predict, inputs="text", outputs="text")
    interface.launch()


if __name__ == "__main__":
    main()
