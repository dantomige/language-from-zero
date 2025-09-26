import unittest
from tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_simple_sentence(self):
        text = "Hello world"
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, ["Hello", "world"])
        self.assertEqual(self.tokenizer.num_tokens(), 2)

    def test_repeated_tokens(self):
        text = "hi hi hi"
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, ["hi", "hi", "hi"])
        self.assertEqual(self.tokenizer.num_tokens(), 1)  # only "hi"

    def test_sentence_with_punctuation(self):
        text = "Hello, world!"
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, ["Hello", ",", "world", "!"])
        self.assertEqual(self.tokenizer.num_tokens(), 4)

    def test_sentence_with_numbers(self):
        text = "I have 2 apples and 10 oranges."
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, ["I", "have", "2", "apples", "and", "10", "oranges", "."])
        self.assertEqual(self.tokenizer.num_tokens(), 8)

    def test_vocab_growth_across_calls(self):
        text1 = "cat dog"
        text2 = "dog mouse"
        self.tokenizer.tokenize(text1)
        self.tokenizer.tokenize(text2)
        vocab = set(self.tokenizer.token_to_id.keys())
        self.assertTrue({"cat", "dog", "mouse"} <= vocab)
        self.assertEqual(self.tokenizer.num_tokens(), 3)

    def test_detokenize(self):
        text = "apple banana"
        tokens = self.tokenizer.tokenize(text)
        ids = [self.tokenizer.token_to_id[t] for t in tokens]
        detok = self.tokenizer.detokenize(ids)
        self.assertEqual(detok, ["apple", "banana"])

    def test_empty_string(self):
        text = ""
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, [])
        self.assertEqual(self.tokenizer.num_tokens(), 0)

if __name__ == "__main__":
    unittest.main()
