from __future__ import annotations
import torch
from pathlib import Path
from src.model.llm import Transformer
from src.tokenizer import Tokenizer
from src.utils.checkpoint import CheckpointManager


class Inference:

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_experiment(
        cls, checkpoint_dir: str | Path, experiment_folder_name: str
    ) -> Inference:
        """Loads the model and tokenizer from a given experiment folder in the checkpoints directory and returns an Inference object.

        Args:
            checkpoint_dir (str | Path): path to checkpoints directory
            experiment_folder_name (str): name of experiment folder in checkpoints directory to load model and tokenizer from

        Returns:
            Inference: Inference object with model and tokenizer loaded from given experiment folder
        """

        checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        bundle = checkpoint_manager.load(experiment_folder_name=experiment_folder_name)

        inference_model = cls(model=bundle.model, tokenizer=bundle.tokenizer)

        return inference_model

    def prepare_query(self, query: str) -> torch.Tensor:
        """Tokenizes the query and converts it to a tensor.

        Args:
            query (str): query to prepare
        Returns:
            torch.Tensor: tensor containing token ids of the query
        """
        query_metadata_start_token_ids = [
            self.tokenizer.get_id(self.tokenizer.START_TOKEN),
            self.tokenizer.get_id(self.tokenizer.HEADER_START_TOKEN),
            self.tokenizer.get_id("user"),
            self.tokenizer.get_id(self.tokenizer.HEADER_END_TOKEN),
        ]
        query_token_ids = self.tokenizer.tokenize(query)
        query_metadata_end_token_ids = [
            self.tokenizer.get_id(self.tokenizer.END_OF_TURN_TOKEN)
        ]
        assistant_response_metadata_token_ids = [
            self.tokenizer.get_id(self.tokenizer.HEADER_START_TOKEN),
            self.tokenizer.get_id("assistant"),
            self.tokenizer.get_id(self.tokenizer.HEADER_END_TOKEN),
        ]

        full_query_token_ids = (
            query_metadata_start_token_ids
            + query_token_ids
            + query_metadata_end_token_ids
            + assistant_response_metadata_token_ids
        )
        ids_tensor = torch.tensor([full_query_token_ids])

        return ids_tensor

    def response(
        self, query: str, max_response_tokens: int, temperature: float = 1.0
    ) -> str:
        """Given a query generates a response by predicting the next token until max_response_tokens is reached.

        Args:
            query (str): query to generate response for
            max_response_tokens (int): maximum number of tokens to generate in response
            temperature (float, optional): controls randomness of predictions. Defaults to 1.0.

        Returns:
            str: generated response
        """
        ids_tensor = self.prepare_query(query)
        ids_in_context_window = ids_tensor[-self.model.context_window :]

        generated_response_token_ids = []

        num_tokens_generated = 0
        last_generated_token_id = None

        while (
            num_tokens_generated < max_response_tokens
            and last_generated_token_id
            != self.tokenizer.get_id(self.tokenizer.END_OF_TURN_TOKEN)
        ):

            # generate the next token id
            new_token_id = self.generate_next_token(
                ids_in_context_window=ids_in_context_window, temperature=temperature
            )

            # update the context window with the new token id
            ids_in_context_window = self.update_window(
                ids_in_context_window=ids_in_context_window, new_token_id=new_token_id
            )

            # append the new token id to the generated response and update the context window
            last_generated_token_id = new_token_id
            num_tokens_generated += 1
            generated_response_token_ids.append(new_token_id)

        return self.tokenizer.decode(generated_response_token_ids)

    def generate_next_token(
        self, ids_in_context_window: torch.Tensor, temperature: float = 1.0
    ) -> int:

        logits = self.model(ids_in_context_window) / temperature
        guesses = self.model.decode(logits)
        new_token_id = guesses[0, -1].item()
        return new_token_id

    def update_window(
        self, ids_in_context_window: torch.Tensor, new_token_id: int
    ) -> torch.Tensor:
        ids_in_context_window = torch.cat(
            [ids_in_context_window, torch.tensor([[new_token_id]])], dim=1
        )
        _, curr_window_size = ids_in_context_window.shape
        if self.model.context_window < curr_window_size:
            ids_in_context_window = ids_in_context_window[:, 1:]

        return ids_in_context_window

    def format(self, response: list[str]) -> str:
        return " ".join(response)


if __name__ == "__main__":
    pass
