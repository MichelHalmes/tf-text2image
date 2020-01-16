
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, GRU, concatenate, Dense,
    LayerNormalization, Reshape, Conv2D, Conv2DTranspose,
    LeakyReLU
)
import tensorflow.keras.backend as Kb



def get_model(encoders):
    chars = Input(shape=[None], name="chars")
    chars_embed = Embedding(input_dim=encoders.chars.vocab_size, output_dim=4)(chars)
    chars_encoded = GRU(16)(chars_embed)

    spec = Input(shape=[None], name="spec")
    spec_embed = Embedding(input_dim=encoders.spec.vocab_size, output_dim=4)(spec)
    spec_encoded = GRU(16)(spec_embed)

    input_encoded = concatenate([chars_encoded, spec_encoded])
    FEAT_HW = 8
    FEAT_C = 32
    features = Dense(FEAT_HW*FEAT_C, activation="selu")(input_encoded)
    features = LayerNormalization()(features)
    features = Dense(FEAT_HW*FEAT_HW*FEAT_C, activation="selu")(features)
    feature_map = Reshape((FEAT_HW, FEAT_HW, FEAT_C))(features)

    feature_map = Conv2DTranspose(filters=32, kernel_size=5, padding="same", strides=2, activation=LeakyReLU(alpha=0.1))(feature_map)
    feature_map = Conv2DTranspose(filters=16, kernel_size=5, padding="same", strides=2, activation=LeakyReLU(alpha=0.1))(feature_map)
    feature_map = Conv2DTranspose(filters=8, kernel_size=5, padding="same", strides=2, activation=LeakyReLU(alpha=0.1))(feature_map)
    image = Conv2DTranspose(filters=6, kernel_size=5, padding="same", strides=2, activation=LeakyReLU(alpha=0.1))(feature_map)
    image = Conv2D(filters=3, kernel_size=3, padding="same", activation="tanh")(image)
    # image =  Kb.clip(image, -1., 1.)
    # We use tanh to help the model saturate the accepted color range.

    model = Model(inputs=[chars, spec], outputs=image)

    return model

# TODO: Build as nested model (ie a class with several Sequentials and a call()) when this is really fixed:
#  https://github.com/tensorflow/tensorflow/issues/21016
