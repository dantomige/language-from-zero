import torch
import 

def train_transformer(model, dataloader, optimizer, criterion):

    model.train()

    for X, y in dataloader:

        optimizer.zero_grad()
        yhat = model.forward(X)

        batch_size, sequence_length = y.shape
        y = y.view(batch_size * sequence_length)

        batch_size, sequence_length, vocab_probs = yhat.shape
        yhat = yhat.view(batch_size * sequence_length, vocab_probs)

        loss = criterion(yhat, y)
        loss.backward
        optimizer.step()


def main():
    # hyperparameters/global parameters

    # load data from hugging face

    # tokenize data and convert to tensor

    # create dataset and dataloader objects

    # create model

    # select optimizer, criteron

    # train model


if __name__ == "__main__":
    pass