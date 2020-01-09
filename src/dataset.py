
import tensorflow_datasets as tfds
# features.chars import TokenTextEncoder, Tokenizer

import config
from data import generate_sample

import tensorflow as tf


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


# def get_encode_text_fn(encoder):
#     # @tf.function
#     def encode(chars, spec, image):
#         print(chars)
#         encoded_chars = encoder.encode(chars.numpy())
#         return encoded_chars, spec, image
#     return encode

def get_generate_fn():
    def generate(_):
        chars, spec, image = generate_sample()
        return chars, spec, image 

    def generate_fn(_):
        return tf.py_function(generate, inp=[_], Tout=(tf.string, tf.string, tf.int32))
    
    return generate_fn


def get_encode_text_fn(chars_encoder, spec_encoder):
    def encode(chars, spec, image):
        encoded_chars = chars_encoder.encode(chars.numpy())
        encoded_spec = spec_encoder.encode(spec.numpy())
        return encoded_chars, encoded_spec, image

    def encode_text_fn(chars, spec, image):
        return tf.py_function(encode, inp=[chars, spec, image], Tout=(tf.int32, tf.int32, tf.int32))
    
    return encode_text_fn


def get_format_img_fn():
    size = [config.IMAGE_H, config.IMAGE_W]
    def format_img(chars, spec, image):
        image = tf.image.resize(image, size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image /= 255.  # Range between 0 and 1
        return chars, spec, image

    def format_img_fn(chars, spec, image):
        return tf.py_function(format_img, inp=[chars, spec, image], Tout=(tf.int32, tf.int32, tf.float32))
    
    return format_img_fn


def get_set_shapes_fn():
    def set_shapes_fn(chars, spec, image):
        chars.set_shape([2])
        spec.set_shape([4])
        image.set_shape([config.IMAGE_H, config.IMAGE_W, 3])
        return {"chars": chars, "spec": spec}, image
    
    return set_shapes_fn


def get_dataset(chars_encoder, spec_encoder):
    SIZE = 128
    # We do all the work in the map functions so that the work can be better paralellized
    ds = tf.data.Dataset.from_generator(
        lambda: range(SIZE), 
        output_types=tf.int32, 
    )
    ds = ds.map(get_generate_fn())
    ds = ds.map(get_encode_text_fn(chars_encoder, spec_encoder))
    ds = ds.map(get_format_img_fn())
    ds = ds.map(get_set_shapes_fn())
    return ds


