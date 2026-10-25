"""Byte-level BPE trained on Python source (whitespace runs become single tokens)."""
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"


def train_bpe(texts, path, vocab_size=16384):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    tok.save(path)
    return CodeTokenizer(path)


class CodeTokenizer:
    def __init__(self, path):
        self.tok = Tokenizer.from_file(path)
        self.eot_token = self.tok.token_to_id(EOT)
        self.vocab_size = self.tok.get_vocab_size()

    def encode_ordinary(self, text):
        return self.tok.encode(text).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids):
        return self.tok.decode([i for i in ids if i != self.eot_token], skip_special_tokens=True)
