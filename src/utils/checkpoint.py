from __future__ import annotations
import json
import torch
from typing import NamedTuple
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime, timezone
from src.utils.io import save_json, read_json
from src.tokenizer import Tokenizer
from src.model.llm import Transformer
from src.config.model import ModelConfig


class ModelBundle(NamedTuple):
    model: Transformer
    tokenizer: Tokenizer
    config: ModelConfig


class ExperimentMetadata(BaseModel):
    experiment_name: str
    time_saved_utc: str
    config_filename: str
    model_state_filename: str
    tokenizer_filename: str

    def save(self, folder_path: str | Path, filename: str):
        path = Path(folder_path) / filename
        save_json(self.model_dump(), path)

    @classmethod
    def load(
        cls, folder_path: str | Path, filename: str
    ) -> ExperimentMetadata:
        path = Path(folder_path) / filename
        metadata_dict = read_json(path)
        return cls.model_validate(metadata_dict)


class CheckpointManager:

    METADATA_FILENAME = "metadata.json"
    CONFIG_FILENAME = "config.json"
    MODEL_STATE_FILENAME = "model_state.pt"
    TOKENIZER_FILENAME = "tokenizer.json"

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)

    def save(
        self,
        experiment_name: str | Path,
        model: Transformer,
        tokenizer: Tokenizer,
        config: ModelConfig,
    ):

        # create experiment directory if it doesn't exist
        folder = self.checkpoint_dir / experiment_name
        folder.mkdir(parents=True, exist_ok=True)

        # save model config
        config.save_pretrained(folder_path=folder, filename=self.CONFIG_FILENAME)

        # save model state
        model.save_pretrained(
            folder_path=folder, model_filename=self.MODEL_STATE_FILENAME
        )

        # save tokenizer state
        tokenizer.save_pretrained(folder_path=folder, filename=self.TOKENIZER_FILENAME)

        # save all in metadata in a json file
        time_saved_utc: datetime = datetime.now(timezone.utc)
        time_saved_utc_formatted: str = time_saved_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

        metadata = ExperimentMetadata(
            experiment_name=experiment_name,
            time_saved_utc=time_saved_utc_formatted,
            config_filename=self.CONFIG_FILENAME,
            model_state_filename=self.MODEL_STATE_FILENAME,
            tokenizer_filename=self.TOKENIZER_FILENAME,
        )

        metadata.save(folder_path=folder, filename=self.METADATA_FILENAME)

        return

    def load(self, experiment_name: str | Path) -> ModelBundle:
        """_summary_

        Args:
            experiment_name (str): _description_

        Returns:
            dict: _description_
        """

        folder_path = self.checkpoint_dir / experiment_name

        metadata = ExperimentMetadata.load(
            folder_path=folder_path, filename=self.METADATA_FILENAME
        )

        # load config
        config = ModelConfig.from_pretrained(
            folder_path=folder_path,
            filename=metadata.config_filename,
        )

        # load model
        model = Transformer.from_pretrained(
            folder_path,
            model_filename=metadata.model_state_filename,
            config_filename=metadata.config_filename,
        )

        # load tokenizer
        tokenizer = Tokenizer.from_pretrained(
            folder_path=folder_path,
            filename=metadata.tokenizer_filename,
        )

        return ModelBundle(model=model, tokenizer=tokenizer, config=config)


if __name__ == "__main__":
    checkpoint_manager = CheckpointManager(checkpoint_dir="src/checkpoints")

    model_config = ModelConfig(
        vocab_size=100,
        d_model=128,
        n_blocks=2,
        n_heads=4,
        context_window=50,
    )

    model = Transformer(config=model_config)
    tokenizer = Tokenizer()

    checkpoint_manager.save(
        experiment_name="test_1",
        model=model,
        tokenizer=tokenizer,
        config=model_config,
    )

    pause = input("Press enter to load checkpoint...")

    bundle = checkpoint_manager.load(experiment_name="test_1")

    print(bundle.config)
    print(bundle.model)
    print(bundle.tokenizer)
