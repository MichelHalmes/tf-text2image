
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, GRU, 
    concatenate, Reshape, Flatten, Dense,
    LayerNormalization, LeakyReLU,
    Conv2D, Conv2DTranspose    
)
import tensorflow.keras.backend as Kb
import tensorflow.keras as K

import config
from utils import CustomSchedule, tanh_cross_entropy


def get_models(encoders):
    text_rnn = _get_text_rnn_model(encoders)
    generator = _get_generator_model(text_rnn)
    discriminator = _get_discriminator_model(text_rnn)
    gan = _get_gan_model(generator, discriminator)

    return generator, discriminator, gan


def _get_text_rnn_model(encoders):
    chars = Input(shape=[None], name="chars")
    chars_embed = Embedding(input_dim=encoders.chars.vocab_size, output_dim=config.EMBED_SIZE)(chars)
    chars_features = GRU(config.RNN_SIZE)(chars_embed)

    spec = Input(shape=[None], name="spec")
    spec_embed = Embedding(input_dim=encoders.spec.vocab_size, output_dim=config.EMBED_SIZE)(spec)
    spec_features = GRU(config.RNN_SIZE)(spec_embed)

    texts_features = concatenate([chars_features, spec_features])

    model = Model(inputs=[chars, spec], outputs=texts_features)
    return model

def _get_generator_model(text_rnn):
    inputs = text_rnn.inputs
    texts_features = text_rnn(inputs)

    FEAT_HW = 8
    FEAT_C = 32
    features = Dense(FEAT_HW*FEAT_C, activation="selu")(texts_features)
    features = LayerNormalization()(features)
    features = Dense(FEAT_HW*FEAT_HW*FEAT_C, activation="selu")(features)
    feature_map = Reshape((FEAT_HW, FEAT_HW, FEAT_C))(features)

    feature_map = Conv2DTranspose(filters=32, kernel_size=5, padding="same", strides=2, activation="relu")(feature_map)
    feature_map = Conv2DTranspose(filters=16, kernel_size=5, padding="same", strides=2, activation="relu")(feature_map)
    feature_map = Conv2DTranspose(filters=8, kernel_size=5, padding="same", strides=2, activation="relu")(feature_map)
    gen_image = Conv2DTranspose(filters=6, kernel_size=5, padding="same", strides=2, activation="relu")(feature_map)
    # We use tanh to help the model saturate the accepted color range.
    gen_image = Conv2D(filters=3, kernel_size=3, padding="same", activation="tanh")(gen_image)
    # gen_image =  Kb.clip(gen_image, -1., 1.)

    model = Model(inputs=inputs, outputs=gen_image)
    optimizer = K.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=tanh_cross_entropy,
                optimizer=optimizer,
                metrics=[K.losses.mean_squared_error, K.losses.mean_absolute_error]
    )
    return model


def _get_discriminator_model(text_rnn):
    text_inputs = text_rnn.inputs
    texts_features = text_rnn(text_inputs)

    image_input = Input(shape=[config.IMAGE_H, config.IMAGE_W, 3], name="image")
    feature_map = Conv2D(filters=8, kernel_size=3, padding="same", strides=2, activation=LeakyReLU(alpha=0.1))(image_input)
    feature_map = Conv2D(filters=16, kernel_size=3, padding="same", strides=2, activation=LeakyReLU(alpha=0.1))(feature_map)
    feature_map = Conv2D(filters=32, kernel_size=3, padding="same", strides=2, activation=LeakyReLU(alpha=0.1))(feature_map)
    feature_map = Conv2D(filters=64, kernel_size=3, padding="same", strides=2, activation=LeakyReLU(alpha=0.1))(feature_map)
    image_features = Flatten()(feature_map)

    features = concatenate([image_features, texts_features])
    features = LayerNormalization()(features)
    fc_hidden = Dense(1024, activation="selu")(features)
    fc_hidden = Dense(128, activation="selu")(fc_hidden)
    p_not_fake = Dense(1, activation="sigmoid")(fc_hidden)


    print(text_inputs)
    print(image_input)
    model = Model(inputs=text_inputs+[image_input], outputs=p_not_fake)
    optimizer = K.optimizers.Adam(learning_rate=0.0002)
    model.compile(loss=K.losses.binary_crossentropy,
                optimizer=optimizer,
                metrics=[K.metrics.binary_accuracy]
    )
    return model

def _get_gan_model(generator, discriminator):
    text_inputs = generator.inputs
    gen_image = generator.output

    discriminator.trainable = False
    p_not_fake = discriminator(text_inputs+[gen_image])

    model = Model(inputs=text_inputs, outputs=p_not_fake)
    optimizer = K.optimizers.Adam(learning_rate=0.0002)
    model.compile(loss=K.losses.binary_crossentropy,
                optimizer=optimizer
    )
    return model


# TODO: Build as nested model (ie a class with several Sequentials and a call()) when this is really fixed:
#  https://github.com/tensorflow/tensorflow/issues/21016
