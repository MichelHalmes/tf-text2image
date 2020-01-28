
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, GRU, 
    concatenate, Reshape, Flatten, Dense,
    LayerNormalization, LeakyReLU, Activation,
    Conv2D, Conv2DTranspose    
)
import tensorflow.keras.backend as Kb
import tensorflow.keras as K
import tensorflow as tf

import config
from utils import CustomSchedule, tanh_cross_entropy, print_model_summary


def get_generator(encoders):
    text_rnn = _get_text_rnn_model(encoders)
    generator = _get_generator_model(text_rnn, gen_trains_rnn=True)

    print_model_summary(generator)

    return text_rnn, generator


def get_models(encoders):
    text_rnn = _get_text_rnn_model(encoders)
    generator = _get_generator_model(text_rnn, gen_trains_rnn=False)
    discriminator = _get_discriminator_model(text_rnn)
    gan = _get_gan_model(generator, discriminator)

    print_model_summary(generator)
    print_model_summary(discriminator)
    print_model_summary(gan)

    return text_rnn, generator, discriminator, gan

def  _get_text_inputs():
    chars = Input(shape=[None], name="chars")
    spec = Input(shape=[None], name="spec")
    return [chars, spec]


def _get_text_rnn_model(encoders):
    chars, spec = _get_text_inputs()

    chars_embed = Embedding(input_dim=encoders.chars.vocab_size, output_dim=config.EMBED_SIZE)(chars)
    chars_features = GRU(config.RNN_SIZE)(chars_embed)

    spec_embed = Embedding(input_dim=encoders.spec.vocab_size, output_dim=config.EMBED_SIZE)(spec)
    spec_features = GRU(config.RNN_SIZE)(spec_embed)

    texts_features = concatenate([chars_features, spec_features])

    model = Model(inputs=[chars, spec], outputs=texts_features, name="text_rnn")
    return model


def _get_generator_model(text_rnn, gen_trains_rnn=False):
    text_inputs = _get_text_inputs()
    texts_features = text_rnn(text_inputs)

    FEAT_HW = 8
    FEAT_C = 32
    def dense(features, n_out):
        features = Dense(n_out, use_bias=False)(features)
        features = LayerNormalization()(features)
        features = Activation("selu")(features)
        return features

    features = dense(texts_features, 4*texts_features.shape[-1])
    features = dense(features, FEAT_HW*FEAT_HW*FEAT_C)
    feature_map = Reshape((FEAT_HW, FEAT_HW, FEAT_C))(features)

    def deconv(feature_map, n_out):
        feature_map = Conv2DTranspose(filters=n_out, kernel_size=5, padding="same", strides=2, use_bias=False)(feature_map)
        feature_map = LayerNormalization()(feature_map)
        feature_map = LeakyReLU(alpha=.1)(feature_map)
        return feature_map

    feature_map = deconv(feature_map, 32)
    feature_map = deconv(feature_map, 16)
    feature_map = deconv(feature_map, 8)
    feature_map = deconv(feature_map, 6)

    # We use tanh to help the model saturate the accepted color range.
    gen_image = Conv2D(filters=3, kernel_size=5, padding="same", activation="tanh")(feature_map)
    assert list(gen_image.shape) == [None, config.IMAGE_H, config.IMAGE_W, 3]

    model = Model(inputs=text_inputs, outputs=gen_image, name="generator")
    # To avoid trivial solutions where the generator manipulates the feature generator,
    # we only allow the disriminator to train the text_rnn
    text_rnn.trainable = False
    model.compile(loss=tanh_cross_entropy,
                optimizer=K.optimizers.Adam(learning_rate=.001),
                metrics=[K.losses.mean_squared_error, K.losses.mean_absolute_error]
    )
    return model


def _get_discriminator_model(text_rnn):
    text_inputs = _get_text_inputs()
    texts_features = text_rnn(text_inputs)

    image_input = Input(shape=[config.IMAGE_H, config.IMAGE_W, 3], name="image")
    def conv(feature_map, n_filters):
        feature_map = Conv2D(filters=n_filters, kernel_size=4, padding="same", strides=2, use_bias=False)(feature_map)
        feature_map = LeakyReLU(alpha=.1)(feature_map)
        return feature_map

    feature_map = conv(image_input, 4)
    feature_map = conv(feature_map, 8)
    feature_map = conv(feature_map, 16)
    feature_map = conv(feature_map, 32)
    feature_map = conv(feature_map, 64)
    image_features = Flatten()(feature_map)

    features = concatenate([image_features, texts_features])
    features = LayerNormalization()(features)
    fc_hidden = Dense(128, activation="selu")(features)
    p_real = Dense(1, activation="sigmoid")(fc_hidden)

    model = Model(inputs=text_inputs+[image_input], outputs=p_real, name="discriminator")
    text_rnn.trainable = False  # TODO
    model.compile(loss=K.losses.binary_crossentropy,
                optimizer=K.optimizers.Adam(learning_rate=.0001, beta_1=.5),
                metrics=[K.metrics.binary_accuracy]
    )
    return model


def _get_gan_model(generator, discriminator):
    text_inputs = _get_text_inputs()
    gen_image = generator(text_inputs)
    p_real = discriminator(text_inputs+[gen_image])
    
    model = Model(inputs=text_inputs, outputs=p_real, name="gan")
    discriminator.trainable = False
    model.compile(loss=K.losses.binary_crossentropy,
                optimizer=K.optimizers.Adam(learning_rate=.0001, beta_1=.5)
    )
    return model


# TODO: Build as nested model (ie a class with several Sequentials and a call()) when this is really fixed:
#  https://github.com/tensorflow/tensorflow/issues/21016
