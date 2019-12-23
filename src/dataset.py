
import tensorflow_datasets as tfds
# features.text import TokenTextEncoder, Tokenizer

import config
from data import iter_data

import tensorflow as tf


class CharTokenizer(object):
    def tokenize(self, s):
        """Splits a string into tokens."""
        s = tf.compat.as_text(s)
        return list(s)
    def join(self, tokens):
        """Joins tokens into a string."""
        return "".join(tokens)


def get_encoder():
    vocab_list = list(config.TEXT_ALPHABETH)
    encoder = tfds.features.text.TokenTextEncoder(
        vocab_list,
        tokenizer=CharTokenizer(),
        decode_token_separator=''
    )
    return encoder

# def get_encode_text_fn(encoder):
#     # @tf.function
#     def encode(text, image):
#         print(text)
#         encoded_text = encoder.encode(text.numpy())
#         return encoded_text, image
#     return encode

def get_encode_text_fn(encoder):
    def encode(text, image):
        encoded_text = encoder.encode(text.numpy())
        return encoded_text, image

    def encode_text_fn(text, image):
        return tf.py_function(encode, inp=[text, image], Tout=(tf.int32, tf.int32))
    
    return encode_text_fn

def get_format_img_fn():
    def format_img(text, image):
        image = tf.image.resize(image, [150, 150])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image /= 255.  # Range between 0 and 1
        return text, image

    def format_img_fn(text, image):
        return tf.py_function(format_img, inp=[text, image], Tout=(tf.int32, tf.float32))
    
    return format_img_fn


def get_dataset(encoder):    
    ds = tf.data.Dataset.from_generator(
        iter_data, 
        output_types=(tf.string, tf.int32), 
    )
    ds = ds.map(get_encode_text_fn(encoder))
    ds = ds.map(get_format_img_fn())
    return ds


