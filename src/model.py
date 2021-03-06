
import tensorflow.keras.backend as Kb
import tensorflow.keras as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, GRU,
    concatenate, Reshape, Flatten, Dense, Lambda, Add,
    LayerNormalization, Dropout, GaussianNoise,
    LeakyReLU, Activation,
    Conv2D, Conv2DTranspose
)

import config as cfg
from utils import CustomLrSchedule, print_model_summary
from losses import tanh_cross_entropy, wasserstein_loss, binary_crossentropy, saturating_binary_crossentropy
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


def _add_layer_variable_prefix(model, prefix):
    def add_prefix(layer):
        layer_conf = layer.get_config()
        layer_conf["name"] = prefix + "/" + layer_conf["name"]
        new_layer = layer.__class__.from_config(layer_conf)
        return new_layer
    return K.models.clone_model(model, clone_function=add_prefix)


def _get_text_inputs():
    chars = Input(shape=[None], name="chars", dtype="int32")
    spec = Input(shape=[None], name="spec")
    return [chars, spec]


def _get_text_rnn_model(encoders):
    chars, spec = _get_text_inputs()
    chars_embed = Embedding(input_dim=encoders.chars.vocab_size, output_dim=cfg.EMBED_SIZE)(chars)
    chars_features = GRU(cfg.RNN_SIZE)(chars_embed)

    spec_embed = Embedding(input_dim=encoders.spec.vocab_size, output_dim=cfg.EMBED_SIZE//2)(spec)
    spec_features = GRU(cfg.RNN_SIZE//2)(spec_embed)

    texts_features = concatenate([chars_features, spec_features])
    texts_features = Dense(4*cfg.RNN_SIZE, use_bias=False)(texts_features)
    texts_features = LayerNormalization()(texts_features)
    texts_features = Activation("selu")(texts_features)

    model = Model(inputs=[chars, spec], outputs=texts_features, name="text_rnn")
    model = _add_layer_variable_prefix(model, "text_rnn")
    return model


def _get_generator_model(text_rnn, gen_trains_rnn):
    text_inputs = _get_text_inputs()
    texts_features = text_rnn(text_inputs)

    def generate_latent(text_inputs):
        shape = [Kb.shape(text_inputs[0])[0], cfg.LATENT_DIM]  # Batch x Dim
        return Kb.random_uniform(shape=shape, minval=-1., maxval=1.)

    latent = Lambda(generate_latent)(text_inputs)

    features = concatenate([texts_features, latent])

    FEAT_HW = 4
    FEAT_C = 32
    features = Dense(FEAT_HW*FEAT_HW*FEAT_C)(features)
    feature_map = Reshape((FEAT_HW, FEAT_HW, FEAT_C))(features)

    def deconv(feature_map, n_out):
        feature_map = Conv2DTranspose(filters=n_out, kernel_size=5, padding="same", strides=2, use_bias=False)(feature_map)
        feature_map = LayerNormalization()(feature_map)
        res_feature_map = LeakyReLU(alpha=.1)(feature_map)
        res_feature_map = Conv2D(filters=n_out, kernel_size=4, padding="same", strides=1)(res_feature_map)
        feature_map = Add()([feature_map, res_feature_map])
        feature_map = LeakyReLU(alpha=.1)(feature_map)
        return feature_map

    feature_map = deconv(feature_map, 32)
    feature_map = deconv(feature_map, 16)

    # We use tanh to help the model saturate the accepted color range.
    gen_image = Conv2DTranspose(filters=3, kernel_size=5, padding="same", strides=2, activation="tanh")(feature_map)
    assert list(gen_image.shape) == [None, cfg.IMAGE_H, cfg.IMAGE_W, 3]

    model = Model(inputs=text_inputs, outputs=gen_image, name="generator")
    # To avoid trivial solutions where the generator manipulates the feature generator,
    # we only allow the disriminator to train the text_rnn
    text_rnn.trainable = gen_trains_rnn
    model.compile(loss=tanh_cross_entropy,
                optimizer=K.optimizers.Adam(learning_rate=cfg.GEN_LR),
                metrics=[K.losses.mean_squared_error, K.losses.mean_absolute_error])
    return model


def _get_discriminator_model(text_rnn):
    text_inputs = _get_text_inputs()
    texts_features = text_rnn(text_inputs)

    def conv(feature_map, n_filters):
        feature_map = Conv2D(filters=n_filters, kernel_size=4, padding="same", strides=2)(feature_map)
        feature_map = Dropout(cfg.DROP_PROB)(feature_map)
        res_feature_map = LeakyReLU(alpha=.1)(feature_map)
        res_feature_map = Conv2D(filters=n_filters, kernel_size=4, padding="same", strides=1)(res_feature_map)
        feature_map = Add()([feature_map, res_feature_map])
        feature_map = LeakyReLU(alpha=.1)(feature_map)
        return feature_map

    image_input = Input(shape=[cfg.IMAGE_H, cfg.IMAGE_W, 3], name="image")
    feature_map = GaussianNoise(cfg.NOISE_VAR)(image_input)
    feature_map = Lambda(lambda img: Kb.clip(img, -1., 1.))(feature_map)
    feature_map = conv(feature_map, 16)  # 16

    texts_feature_map = Dense(16*16*4)(texts_features)
    texts_feature_map = Reshape((16, 16, 4))(texts_feature_map)
    feature_map = concatenate([feature_map, texts_feature_map])

    feature_map = conv(feature_map, 32)  # 8
    feature_map = conv(feature_map, 64)  # 4

    feature_map = Conv2D(filters=32, kernel_size=1, padding="same", strides=1, activation=LeakyReLU(alpha=.1))(feature_map)
    feature_map = Dropout(cfg.DROP_PROB)(feature_map)

    image_features = Flatten()(feature_map)
    features = concatenate([image_features, texts_features])
    if cfg.USE_MBD:
        minibatch_features = MinibatchDiscrimination(num_kernels=cfg.MBD_KERNELS, dim_per_kernel=cfg.MBD_DIMS)(image_features)
        features = concatenate([features, minibatch_features])

    logits_p_real = Dense(1, use_bias=not cfg.USE_WGAN_GP)(features)

    model = Model(inputs=text_inputs+[image_input], outputs=logits_p_real, name="discriminator")
    text_rnn.trainable = True
    model.compile(loss=wasserstein_loss if cfg.USE_WGAN_GP else binary_crossentropy,
                optimizer=K.optimizers.Adam(learning_rate=CustomLrSchedule(), beta_1=cfg.DIS_BETA_1, beta_2=cfg.DIS_BETA_2),
                metrics=[K.metrics.BinaryAccuracy(threshold=.0)])  # Since we output logits, threshold .0 corresponds to .5 on the sigmoid

    return model


def _get_gan_model(generator, discriminator):
    text_inputs = _get_text_inputs()
    gen_image = generator(text_inputs)
    logits_p_real = discriminator(text_inputs+[gen_image])

    model = Model(inputs=text_inputs, outputs=logits_p_real, name="gan")
    discriminator.trainable = False
    model.compile(loss=wasserstein_loss if cfg.USE_WGAN_GP else binary_crossentropy,
                optimizer=K.optimizers.Adam(learning_rate=CustomLrSchedule(), beta_1=cfg.DIS_BETA_1, beta_2=cfg.DIS_BETA_2))
    return model
