# language-from-zero
A large language model from scratch

-figure out how to train the model
-adding padding and other special tokens
-gradio for a quick gui
-continuously building the output through appending guessed token
-acquiring a lot of training data and potentially training it on specific topics
-add special tokens to data (bos and eos and unk) appropriate (by chapter, by book, etc.) in data
-account for padding in cross entropy and vector embedding layer and attention (padding mask)
-figure out how to train model faster
-build inference class
    -my guess is that it takes the user input and starts building a response based on the context from that and then adds one by one until it his the context window size of eos if it hits the context window size then it simply shifts the window to the right and then adds a new token so llm loses a bit of information for longer sequences.
    -improve inference with key value cache
    -start with a small number of tokens and build up to the context window used it training (should NOT pass this length)

-connect inference to ui
-make sure vocab is correct: contains punctuations, deals with contractions, etx.

DATA AND TRAINNG
-the dataloader is concerned with loading in the data
-dataset class as an abstraction of interacting with the data (for example the context window) acting as an adapter
-seperate training function to specify how to train the model
-creating a response from the model consists (creating an inference wrapper)
-making sure model weights that should be trained are getting trained, ones that shouldnt are not
-types of tokens are sos, eos, pad, and STOP (and implementing to both tokenizer and getting the model to actually output this)

-training bigger models, faster, better (train on podcasts, reddit, and other conversational things sos, sep, eos)
    -cut off training before overfitting (LOL WE NOT EVEN CLOSE TO OVERFITTING BRODIE)
    -have a method to view accuracy of model on a test set of data
    -more effectively calibrate vocab size to model capacity
    -train on special tokens
    -fetch better training data
-correcting special tokens, ensure parsing, and sizeable vocab (eos, sos, sep, unk, pad)
-multihead attention
    -customizable ffn in each layer
-autotraining with user interaction?
-organize file structure, make sure variable names are consistent, add type hints and doc string, seperation of concern, encapsulation, and other styling/practices
    -convert save runs to model state dict, tokenizer state dict, and config
    -make it so training runs automatically land in an experiments folder and create a folder for containing the things above