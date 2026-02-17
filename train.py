import torch
import torch.nn as nn
from utils import save_as_json
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets import load_dataset
from tokenizer import Tokenizer
from llm import Transformer
from src.dataset import LangaugeModelDataset


def train_transformer(model, dataloader, optimizer, criterion, device, num_datapoints_to_train = None):

    model.train()
    model.to(device)

    num_examples_per_batch = dataloader.batch_size

    for batch_idx, (X, y) in enumerate(dataloader):
        num_trained = num_examples_per_batch * batch_idx
        if num_datapoints_to_train is not None and num_trained > num_datapoints_to_train:
            break

        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        yhat = model(X)

        batch_size, sequence_length = y.shape
        y = y.view(batch_size * sequence_length)

        batch_size, sequence_length, vocab_probs = yhat.shape
        yhat = yhat.view(batch_size * sequence_length, vocab_probs)

        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx} with training loss: ", loss)


def main():
    # hyperparameters/global parameters
    print("Hyperparameters")
    d_model = 64
    context_window = 128
    batch_size = 8
    n_blocks = 6
    print(d_model, context_window)

    # load data from hugging face
    print("Loading dataset dataset ...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n".join(dataset["text"])
    print(type(full_text))

    # tokenize data and convert to tensor
    print("Tokenizing")
    # special_tokens = {
    #     "<PAD>": 0,
    #     "<BOS>": 1,
    #     "<EOS>": 2,
    #     "<UNK>": 3,
    #     }
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(full_text)
    tokens_tensor = torch.tensor(tokens)
    vocab_size = tokenizer.vocab_size
    print("Vocab size: ", vocab_size)
    print("Token tensor shape: ", tokens_tensor.shape)

    # create dataset and dataloader objects
    print("Creating dataset and dataloader objects")
    dataset = LangaugeModelDataset(tokens=tokens_tensor, context_window=context_window)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    num_datapoints_to_train = 1000
    print("Dataloader length: ", len(dataloader))

    # create model
    print("Creating model")
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_blocks=n_blocks,
        max_seq_len=context_window,
    )

    # select optimizer, criterion
    print("Selecting optimizer and criterion")
    optimizer = Adam(model.parameters())
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
        num_datapoints_to_train=num_datapoints_to_train
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
        "context_window": context_window,
        "utc_time": utc_time_saved,
        "tokenizer_filename": tokenizer_filename,
    }
    torch.save(checkpoint, f"checkpoint_{utc_time_saved}.pth")


if __name__ == "__main__":
    main()
