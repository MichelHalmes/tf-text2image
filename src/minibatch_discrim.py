import tensorflow.keras as K
import tensorflow.keras.backend as Kb

import config

class MinibatchDiscrimination(K.layers.Layer):

    def __init__(self, num_kernels, dim_per_kernel, **kwargs):
        super().__init__(**kwargs)
        self._num_kernels = num_kernels  # `B` in the paper
        self._dim_per_kernel = dim_per_kernel  # `C` in the paper

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]  # `A` in the paper

        self._T = self.add_weight(shape=(self._num_kernels, input_dim, self._dim_per_kernel),
            initializer="lecun_normal",
            name="mbd_kernel",
            trainable=True)

        super().build(input_shape)

    def call(self, x, mask=None):
        activation = Kb.dot(x, self._T)  # (batch, in_dim) x (nb_ker, in_dim, ker_dim) = (batch, nb_ker, ker_dim) 
        diffs = (Kb.expand_dims(activation, 3)  # (batch, nb_ker, ker_dim, 1)
                    - Kb.expand_dims(Kb.permute_dimensions(activation, [1, 2, 0]), 0))  # (1, nb_ker, ker_dim, batch)
        abs_diffs = Kb.sum(Kb.abs(diffs), axis=2)  # (batch, nb_ker, batch) Sum over rows to get L1
        minibatch_features = Kb.sum(Kb.exp(-abs_diffs), axis=2)  # (batch, nb_ker)
        return minibatch_features # (batch, in_dim+nb_ker)

