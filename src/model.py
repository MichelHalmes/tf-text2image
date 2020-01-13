
import tensorflow.keras as K



def get_model(chars_encoder, spec_encoder):
    chars = K.layers.Input(shape=[None], name="chars")
    chars_embed = K.layers.Embedding(input_dim=chars_encoder.vocab_size, output_dim=4)(chars)
    chars_encoded = K.layers.GRU(16)(chars_embed)

    spec = K.layers.Input(shape=[None], name="spec")
    spec_embed = K.layers.Embedding(input_dim=spec_encoder.vocab_size, output_dim=4)(spec)
    spec_encoded = K.layers.GRU(16)(spec_embed)

    input_encoded = K.layers.concatenate([chars_encoded, spec_encoded])
    FEAT_HW = 8
    FEAT_C = 32
    features = K.layers.Dense(FEAT_HW*FEAT_C, activation="selu")(input_encoded)
    features = K.layers.LayerNormalization()(features)
    features = K.layers.Dense(FEAT_HW*FEAT_HW*FEAT_C, activation="selu")(features)
    feature_map = K.layers.Reshape((FEAT_HW, FEAT_HW, FEAT_C))(features)

    feature_map = K.layers.Conv2DTranspose(filters=32, kernel_size=3, padding="same", strides=2, activation="relu")(feature_map)
    feature_map = K.layers.Conv2DTranspose(filters=16, kernel_size=3, padding="same", strides=2, activation="relu")(feature_map)
    feature_map = K.layers.Conv2DTranspose(filters=8, kernel_size=3, padding="same", strides=2, activation="relu")(feature_map)
    feature_map = K.layers.Conv2DTranspose(filters=6, kernel_size=3, padding="same", strides=2, activation="relu")(feature_map)
    image = K.layers.Conv2D(filters=3, kernel_size=5, padding="same", activation="relu")(feature_map)

    model = K.models.Model(inputs=[chars, spec], outputs=image)

    return model


# class MyModel(K.models.Model):
#     # TODO: Build as nested model when this is really fixed: https://github.com/tensorflow/tensorflow/issues/21016

#     def __init__(self, encoder):
#         super().__init__()
#         self._rnn = self._init_rnn(encoder.vocab_size)
#         self._fc = self._init_fc()
#         self._deconv = self._init_deconv()

#     def _init_rnn(self, size):
#         rnn = K.Sequential()
#         rnn.add(K.layers.Embedding(input_dim=size, output_dim=4))
#         rnn.add(K.layers.GRU(16))
#         return rnn

#     def _init_fc(self):
#         FEAT_HW = 8
#         FEAT_C = 8
#         fc = K.Sequential()
#         fc.add(K.layers.Dense(FEAT_HW*FEAT_HW*FEAT_C, activation="relu"))
#         fc.add(K.layers.Reshape((FEAT_HW, FEAT_HW, FEAT_C)))
#         return fc

#     def _init_deconv(self):
#         deconv = K.Sequential()
#         deconv.add(K.layers.Conv2DTranspose(filters=6, kernel_size=3, padding="same", strides=2, activation="relu"))
#         deconv.add(K.layers.Conv2DTranspose(filters=6, kernel_size=3, padding="same", strides=2, activation="relu"))
#         deconv.add(K.layers.Conv2DTranspose(filters=4, kernel_size=3, padding="same", strides=2, activation="relu"))
#         deconv.add(K.layers.Conv2DTranspose(filters=4, kernel_size=3, padding="same", strides=2, activation="relu"))
#         deconv.add(K.layers.Conv2D(filters=3, kernel_size=2, padding="same", activation="relu"))

#         return deconv

#     def call(self, text):
#         encoding = self._rnn(text)
#         feture_map = self._fc(encoding)
#         image = self._deconv(feture_map)
#         return image

