import torch
import torch.nn as nn
from typing import Optional
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from datasets import load_dataset
from src.utils import save_as_json
from src.tokenizer import Tokenizer
from src.model.llm import Transformer
from src.data.dataset import LangaugeModelDataset


def train_transformer(
    model: Transformer,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 1,
    num_batches: Optional[int] = None,
    stop_loss: Optional[int] = None
):

    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        for batch_idx, (X, y) in enumerate(dataloader):
            if num_batches is not None and num_batches == batch_idx:
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


def main():
    # hyperparameters/global parameters
    print("Hyperparameters")
    d_model = 256
    context_window = 256
    batch_size = 16
    n_blocks = 6
    n_heads = 8
    print(d_model, context_window)

    # load data from hugging face
    print("Loading dataset dataset ...")
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # print(type(dataset), len(dataset))
    # full_text = "\n".join(dataset["text"])

    # shuffled_dataset = dataset.shuffle(seed=17)
    # shuffled_full_text = "\n".join(shuffled_dataset["text"])

    # dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    # medium_dataset = dataset.take(1000)
    # # print(type(medium_dataset), len(medium_dataset))
    # full_text = " ".join(story["text"] for story in medium_dataset)

    # print(type(full_text), len(full_text))
    # print(type(shuffled_full_text), len(shuffled_full_text))

    # 1. Load the Gutenberg dataset (ungated, streaming)
    dataset = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)

    # 2. Take the first few books (even 10 books is millions of characters)
    # Gutenberg books are HUGE, so .take(10) is plenty for a 400k sample
    tokenizer_dataset = dataset.skip(10).take(3)
    training_dataset = dataset.skip(10).take(1)

    # first_book = next(iter(medium_dataset))
    # print(first_book.keys())

    # 3. Join them into one giant 'Stream of Consciousness' string
    tokenizer_full_text = " ".join(book["TEXT"] for book in tokenizer_dataset)
    training_full_text = " ".join(book["TEXT"] for book in training_dataset)

    print("Tokening")
    print(f"Type: {type(tokenizer_full_text)}")
    print(f"Total Characters: {len(tokenizer_full_text):,}")
    print(f"Sample: {tokenizer_full_text[:300]}")

    print("Training")
    print(f"Type: {type(training_full_text)}")
    print(f"Total Characters: {len(training_full_text):,}")
    print(f"Sample: {training_full_text[:300]}")

    update_vocab_sample_size = 4_000_000

    # dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    # print(list(dataset.items()))
    # full_text = ""

    # tokenize data and convert to tensor
    print("Tokenizing")
    tokenizer = Tokenizer()
    tokenizer.tokenize(tokenizer_full_text, update_vocab=True)
    # tokenizer.tokenize(shuffled_full_text[:update_vocab_sample_size], update_vocab=True)
    tokens = tokenizer.tokenize(training_full_text)
    tokens_tensor = torch.tensor(tokens)
    vocab_size = tokenizer.vocab_size
    print("Vocab size: ", vocab_size)
    print("Token tensor shape: ", tokens_tensor.shape)
    # print("Token to id: ", tokenizer.token_to_id)

    # create dataset and dataloader objects
    print("Creating dataset and dataloader objects")
    dataset = LangaugeModelDataset(tokens=tokens_tensor, context_window=context_window)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    num_epochs = 2
    num_batches = None
    stop_loss = 2.1

    print("Dataloader length: ", len(dataloader))

    # create model
    print("Creating model")
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=n_heads,
        context_window=context_window,
    )

    # select optimizer, criterion
    print("Selecting optimizer and criterion")
    learning_rate = 1e-4
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
        num_batches=num_batches,
        stop_loss=stop_loss
    )

    print("Saving model")
    utc_time_saved = datetime.now(timezone.utc)
    tokenizer_filename = f"tokenizer_{utc_time_saved}"
    tokenizer_state_dict = tokenizer.state_dict()
    tokenizer = save_as_json(tokenizer_state_dict, tokenizer_filename)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_blocks": n_blocks,
        "n_heads": n_heads,
        "context_window": context_window,
        "utc_time": utc_time_saved,
        "tokenizer_filename": tokenizer_filename,
    }
    torch.save(checkpoint, f"checkpoint_{utc_time_saved}.pth")


if __name__ == "__main__":
    main()
