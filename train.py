import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from src.tokenizer import Tokenizer
from src.config.model import ModelConfig
from src.model.llm import Transformer
from src.utils.checkpoint import CheckpointManager
from src.data.processor import DataProcessor


def train_transformer(
    model: Transformer,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 1,
    num_batches_per_epoch: Optional[int] = None,
    stop_loss: Optional[int] = None,
):

    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        for batch_idx, (X, y) in enumerate(dataloader):
            if num_batches_per_epoch is not None and num_batches_per_epoch == batch_idx:
                continue

            optimizer.zero_grad()

            X: torch.Tensor = X.to(device)
            y: torch.Tensor = y.to(device)
            yhat: torch.Tensor = model(X)

            batch_size, sequence_length = y.shape
            y = y.view(batch_size * sequence_length)

            batch_size, sequence_length, vocab_probs = yhat.shape
            yhat = yhat.view(batch_size * sequence_length, vocab_probs)

            loss: torch.Tensor = criterion(yhat, y)
            loss.backward()
            optimizer.step()

            print(f"Batch {batch_idx} with training loss: ", loss)

            if stop_loss is not None and loss < stop_loss:
                return
        print(f"Epoch {epoch} with training loss: ", loss)


def train_tokenizer(tokenizer: Tokenizer, text: str):
    tokenizer.tokenize(text, update_vocab=True)

def main():

    # hyperparameters/global parameters
    print("Hyperparameters")
    d_model = 256
    context_window = 256
    batch_size = 16
    n_blocks = 6
    n_heads = 8
    print("d_model: ", d_model)
    print("context_window: ", context_window)
    print("batch_size: ", batch_size)
    print("n_blocks: ", n_blocks)
    print("n_heads: ", n_heads)

    # load data and create dataset and dataloader objects
    print("Loading dataset and creating dataloader ...")
    tokenizer = Tokenizer.from_pretrained(
        folder_path="src/checkpoints/head", filename="tokenizer.json"
    )
    data_processor = DataProcessor(tokenizer=tokenizer, context_window=context_window)
    dataset = data_processor.from_nltk_gutenberg()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    print("Number of batches in dataloader: ", len(dataloader))

    # create model
    print("Creating model")
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=n_heads,
        context_window=context_window,
    )
    model = Transformer(config=config)

    # select optimizer, criterion
    print("Selecting optimizer and criterion")
    learning_rate = 1e-4
    num_epochs = 2
    num_batches_per_epoch = None
    stop_loss = None
    print("Learning rate: ", learning_rate)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # train model
    print("Training model ...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    train_transformer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        num_batches_per_epoch=num_batches_per_epoch,
        stop_loss=stop_loss,
    )

    # save model and tokenizer
    print("Saving model")
    checkpoint_manager = CheckpointManager(checkpoint_dir="/src/checkpoints")
    checkpoint_manager.save(
        "gutenburg_transformer", model=model, tokenizer=tokenizer, config=config
    )


if __name__ == "__main__":
    main()
