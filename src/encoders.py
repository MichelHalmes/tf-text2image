
import tensorflow_datasets as tfds
# features.chars import TokenTextEncoder, Tokenizer
import tensorflow as tf

import config
from data import generate_sample



class CharTokenizer(object):
    def tokenize(self, s):
        """Splits a string into tokens."""
        s = tf.compat.as_text(s)
        return list(s)
    def join(self, tokens):
        """Joins tokens into a string."""
        return "".join(tokens)


def get_chars_encoder():
    vocab_list = list(config.CHARS_ALPHABETH)
    chars_encoder = tfds.features.text.TokenTextEncoder(
        vocab_list,
        tokenizer=CharTokenizer(),
        decode_token_separator=''
    )
    return chars_encoder

def get_spec_encoder():
    tokenizer = tfds.features.text.Tokenizer()

    vocabulary_set = set()
    no_update_cnt = 0
    while no_update_cnt < 100:
        _, spec, _ = generate_sample()
        tokens = tokenizer.tokenize(spec)
        if tokens:
            no_update_cnt += 1
            vocabulary_set.update(tokens)

    vocab_list = sorted(vocabulary_set)
    spec_encoder = tfds.features.text.TokenTextEncoder(
        vocab_list,
        tokenizer=tokenizer)
    return spec_encoder

