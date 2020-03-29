import random

import tensorflow as tf
# from tensorflow.data.experimental import AUTOTUNE

import config
from data.generate import generate_sample


def get_generate_fn(difficulty):
    def generate(_):
        chars, spec, image = generate_sample(difficulty)
        # chars += config.EOS_CHAR
        # spec += " "+config.EOS_CHAR
        # TODO: Retry with EOS for variable length inputs
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


def get_format_img_fn(img_encoder):
    size = [config.IMAGE_H, config.IMAGE_W]

    def format_img(chars, spec, image):
        image = tf.image.resize(image, size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image /= 255.  # Range between 0. and 1.
        image = img_encoder.encode(image)  # normalize

        return chars, spec, image

    def format_img_fn(chars, spec, image):
        # TODO: avoid py_function
        return tf.py_function(format_img, inp=[chars, spec, image], Tout=(tf.int32, tf.int32, tf.float32))

    return format_img_fn


def get_set_shapes_fn():
    def set_shapes_fn(chars, spec, image):
        chars.set_shape([2])
        spec.set_shape([4])
        image.set_shape([config.IMAGE_H, config.IMAGE_W, 3])
        return {"chars": chars, "spec": spec}, image

    return set_shapes_fn


def _iter_forever():
    cnt = 0
    while True:
        cnt += 1
        yield cnt


def get_dataset(encoders, difficulty=-1):
    # We do all the work in the map functions so that the work can be better paralellized
    ds = tf.data.Dataset.from_generator(
        _iter_forever,
        output_types=tf.int32,
    )
    ds = ds.map(get_generate_fn(difficulty))
    ds = ds.map(get_encode_text_fn(encoders.chars, encoders.spec))
    ds = ds.map(get_format_img_fn(encoders.image))
    ds = ds.map(get_set_shapes_fn())
    ds = ds.prefetch(config.BATCH_SIZE * 2)
    return ds
