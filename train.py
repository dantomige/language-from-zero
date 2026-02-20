import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader, random_split
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
                break

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


def validate_transformer(
    model: Transformer,
    validate_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_batches: Optional[int] = None,
):

    model.eval()
    model.to(device)

    total_loss = 0.0
    num_batches_processed = 0

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(validate_dataloader):
            if num_batches is not None and num_batches == batch_idx:
                break

            X: torch.Tensor = X.to(device)
            y: torch.Tensor = y.to(device)
            yhat: torch.Tensor = model(X)

            batch_size, sequence_length = y.shape
            y = y.view(batch_size * sequence_length)

            batch_size, sequence_length, vocab_probs = yhat.shape
            yhat = yhat.view(batch_size * sequence_length, vocab_probs)

            loss: torch.Tensor = criterion(yhat, y)
            total_loss += loss.item()
            num_batches_processed += 1

    avg_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0.0
    return avg_loss


def train_transformer_with_validation(
    model: Transformer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 1,
    num_batches_per_epoch: Optional[int] = None,
    num_validation_batches: Optional[int] = None,
    stop_loss: Optional[int] = None,
):

    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        for batch_idx, (X, y) in enumerate(train_dataloader):
            if num_batches_per_epoch is not None and num_batches_per_epoch == batch_idx:
                break

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

        validation_loss = validate_transformer(
            model=model,
            validate_dataloader=validation_dataloader,
            criterion=criterion,
            device=device,
            num_batches=num_validation_batches,
        )

        print(f"Epoch {epoch} with training loss: ", loss)
        print(f"Epoch {epoch} with validation loss: ", validation_loss)


def train_nlkt_gutenberg():

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
    tokenizer = Tokenizer()
    data_processor = DataProcessor(tokenizer=tokenizer, context_window=context_window)
    dataset = data_processor.from_nltk_gutenberg(update_vocab=True)
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
    num_epochs = 5
    num_batches_per_epoch = 500
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


def train_shareGPT():

    # hyperparameters/global parameters
    print("Hyperparameters")
    d_model = 512
    context_window = 512
    batch_size = 8
    n_blocks = 6
    n_heads = 8
    print("d_model: ", d_model)
    print("context_window: ", context_window)
    print("batch_size: ", batch_size)
    print("n_blocks: ", n_blocks)
    print("n_heads: ", n_heads)

    # load data and create dataset and dataloader objects
    print("Loading dataset and creating dataloader ...")
    tokenizer = Tokenizer()
    data_processor = DataProcessor(tokenizer=tokenizer, context_window=context_window)

    tokenizer.add_to_vocab("user")
    tokenizer.add_to_vocab("assistant")
    data_processor.from_nltk_gutenberg(
        num_books=5, update_vocab=True
    )  # prime the tokenizer with some words from the gutenberg dataset, to speed up tokenization of shareGPT conversations

    dataset = data_processor.from_shareGPT(limit=150)
    train_size = int(0.9 * len(dataset))
    validate_size = len(dataset) - train_size
    train_dataset, validate_dataset = random_split(
        dataset,
        [train_size, validate_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    validate_dataloader = DataLoader(
        dataset=validate_dataset, batch_size=batch_size, shuffle=False
    )

    print("Number of batches in train dataloader: ", len(train_dataloader))
    print("Number of batches in validate dataloader: ", len(validate_dataloader))
    print("Vocab size: ", tokenizer.vocab_size)

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
    weight_decay = 1e-2
    num_epochs = 3
    num_batches_per_epoch = 2500
    num_validation_batches = 10
    stop_loss = None
    print("Learning rate: ", learning_rate)
    print("Weight decay: ", weight_decay)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    label_smoothing = 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # train model
    print("Training model ...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    train_transformer_with_validation(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validate_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        num_batches_per_epoch=num_batches_per_epoch,
        num_validation_batches=num_validation_batches,
        stop_loss=stop_loss,
    )

    # save model and tokenizer
    print("Saving model")
    checkpoint_manager = CheckpointManager(checkpoint_dir="src/checkpoints")
    checkpoint_manager.save(
        "medium_shareGPT_transformer_v2_with_validation_and_regularization",
        model=model,
        tokenizer=tokenizer,
        config=config,
    )


def main():
    # train_nlkt_gutenberg()
    train_shareGPT()


if __name__ == "__main__":
    main()
