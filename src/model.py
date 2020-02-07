
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, GRU, 
    concatenate, Reshape, Flatten, Dense,
    LayerNormalization, BatchNormalization, Dropout,
    LeakyReLU, Activation,
    Conv2D, Conv2DTranspose    
)
import tensorflow.keras.backend as Kb
import tensorflow.keras as K
import tensorflow as tf

import config
from utils import CustomSchedule, print_model_summary
from losses import tanh_cross_entropy, wasserstein_loss, binary_crossentropy, alternative_binary_crossentropy
from minibatch_discrim import MinibatchDiscrimination

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

    spec_embed = Embedding(input_dim=encoders.spec.vocab_size, output_dim=config.EMBED_SIZE//2)(spec)
    spec_features = GRU(config.RNN_SIZE//2)(spec_embed)

    texts_features = concatenate([chars_features, spec_features])
    texts_features = Dense(4*config.RNN_SIZE, use_bias=False)(texts_features)
    texts_features = LayerNormalization()(texts_features)
    texts_features = Activation("selu")(texts_features)

    model = Model(inputs=[chars, spec], outputs=texts_features, name="text_rnn")
    return model


def _get_generator_model(text_rnn, gen_trains_rnn):
    text_inputs = _get_text_inputs()
    texts_features = text_rnn(text_inputs)

    FEAT_HW = 4
    FEAT_C = 32
    features = Dense(FEAT_HW*FEAT_HW*FEAT_C)(texts_features)
    feature_map = Reshape((FEAT_HW, FEAT_HW, FEAT_C))(features)

    def deconv(feature_map, n_out):
        feature_map = Conv2DTranspose(filters=n_out, kernel_size=5, padding="same", strides=2, use_bias=False)(feature_map)
        feature_map = LayerNormalization()(feature_map)
        feature_map = LeakyReLU(alpha=.1)(feature_map)
        return feature_map

    feature_map = deconv(feature_map, 32)
    feature_map = deconv(feature_map, 16)

    # We use tanh to help the model saturate the accepted color range.
    gen_image = Conv2DTranspose(filters=3, kernel_size=5, padding="same", strides=2, activation="tanh")(feature_map)
    assert list(gen_image.shape) == [None, config.IMAGE_H, config.IMAGE_W, 3]

    model = Model(inputs=text_inputs, outputs=gen_image, name="generator")
    # To avoid trivial solutions where the generator manipulates the feature generator,
    # we only allow the disriminator to train the text_rnn
    text_rnn.trainable = gen_trains_rnn
    model.compile(loss=tanh_cross_entropy,
                optimizer=K.optimizers.Adam(learning_rate=config.GEN_LR),
                metrics=[K.losses.mean_squared_error, K.losses.mean_absolute_error]
    )
    return model


def _get_discriminator_model(text_rnn):
    text_inputs = _get_text_inputs()
    texts_features = text_rnn(text_inputs)

    def conv(feature_map, n_filters):
        feature_map = Conv2D(filters=n_filters, kernel_size=4, padding="same", strides=2)(feature_map)
        feature_map = LeakyReLU(alpha=.1)(feature_map)
        feature_map = Dropout(.3)(feature_map)
        return feature_map

    image_input = Input(shape=[config.IMAGE_H, config.IMAGE_W, 3], name="image")

    feature_map = image_input
    feature_map = conv(feature_map, 16)  #16

    texts_feature_map = Dense(16*16*4)(texts_features)
    texts_feature_map = Reshape((16, 16, 4))(texts_feature_map)
    feature_map = concatenate([feature_map, texts_feature_map])

    feature_map = conv(feature_map, 32)  #8
    feature_map = conv(feature_map, 64)  #4

    feature_map = Conv2D(filters=32, kernel_size=1, padding="same", strides=1, activation=LeakyReLU(alpha=.1))(feature_map)

    image_features = Flatten()(feature_map)
    minibatch_features = MinibatchDiscrimination(num_kernels=20)(image_features)

    features = concatenate([image_features, minibatch_features, texts_features])
    logits_p_real = Dense(1)(features)

    model = Model(inputs=text_inputs+[image_input], outputs=logits_p_real, name="discriminator")
    text_rnn.trainable = False  # TODO
    model.compile(loss=binary_crossentropy,
                optimizer=K.optimizers.Adam(learning_rate=config.DIS_LR, beta_1=config.DIS_BETA_1),
                metrics=[K.metrics.BinaryAccuracy(threshold=.0)]  # Since we output logits, threshold .0 corresponds to .5 on the sigmoid
    )
    return model


def _get_gan_model(generator, discriminator):
    text_inputs = _get_text_inputs()
    gen_image = generator(text_inputs)
    logits_p_real = discriminator(text_inputs+[gen_image])
    
    model = Model(inputs=text_inputs, outputs=logits_p_real, name="gan")
    discriminator.trainable = False
    model.compile(loss=binary_crossentropy,
                optimizer=K.optimizers.Adam(learning_rate=config.DIS_LR, beta_1=config.DIS_BETA_1)
    )
    return model


# TODO: Build as nested model (ie a class with several Sequentials and a call()) when this is really fixed:
#  https://github.com/tensorflow/tensorflow/issues/21016
