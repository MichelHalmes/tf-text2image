import tensorflow.keras.backend as Kb
import tensorflow.keras as K
import tensorflow as tf
import numpy as np

import config as cfg

##   TANH CROSS ENTROPY   ##

def tanh_cross_entropy(y_true, y_pred):
    # See README for details 
    use_logits, logits = _get_tanh_logits(y_pred)
    if use_logits:
        # We use the numerically stable method from logits
        return _tce_logit_abs(y_true, logits)
    else:
        # logging.warn("Caution using tce-loss without logits") TODO: reactivate
        return _tce(y_true, y_pred)


def _get_tanh_logits(y_pred):
    """ Tries to recover the logits from the opartion to use stable method """
    if tf.executing_eagerly():
        return False, None
    elif y_pred.op.type == "Tanh":
        assert len(y_pred.op.inputs) == 1
        logits = y_pred.op.inputs[0]
        return True, logits
    elif y_pred.op.type == "Identity":
        assert len(y_pred.op.inputs) == 1
        y_pred = y_pred.op.inputs[0]
        return _get_tanh_logits(y_pred)
    else:
        return False, None


def _tce_logit_abs(y_true, logits):
        # We always have negative terms in the exp, making it numerically stable
        term = 1. + Kb.exp(-2.*Kb.abs(logits))
        tce = Kb.log(term) + Kb.abs(logits)
        tce -= y_true*logits + Kb.log(2.)
        return Kb.mean(tce, axis=-1)


def _tce(y_true, y_pred):    
        y_pred = Kb.clip(y_pred, -1., 1.)
        tce = (1. - y_true) * Kb.log(1. - y_pred + Kb.epsilon())
        tce += (1. + y_true) * Kb.log(1. + y_pred + Kb.epsilon())
        return -.5 * Kb.mean(tce, axis=-1)

##   WASSERSTEIN LOSS  ##

def wasserstein_loss(y_true, y_pred):
    """ y_pred: ``logit``=f in the paper
        y_true: (fake, real) = (0, 1)
            should be mapped to the multipliers for f: (1, -1)
    """
    multiplier = y_true*-2. + 1.
    return Kb.mean(multiplier * y_pred, axis=-1)


def clip_weights(network, clip_value):
    """ Used in combination with Wasserstein loss: https://arxiv.org/abs/1701.07875 """
    for layer in network.layers:
        weights = layer.get_weights()
        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
        layer.set_weights(weights)

class GradientPenalizer(object):
    """ Implementation of: https://arxiv.org/abs/1704.00028
        This is the most elegant solution I could come up with:
        https://github.com/tensorflow/tensorflow/issues/36436
    """

    def __init__(self, discriminator, gp_only):
        """ gp_only: 
                True -> Minimizes the gradient penalty only
                False -> Also optimizes our estimate of the Wasserstein distance """
        self._discriminator = discriminator
        self._gp_only = gp_only
        self.name = "gradient_penalizer"
        self.metrics_names = ["gp", "wdist"]

    # Cannot be a tf.function since we declare a variable... :-(
    def run_on_batch(self, text_inputs, real_images, fake_images):
        interpolated_images = self._interpolate_images(real_images, fake_images)
        return self.run_step(text_inputs, real_images, fake_images, interpolated_images)

    def _interpolate_images(self, real_images, fake_images):
        eps = Kb.random_uniform([cfg.BATCH_SIZE, 1, 1, 1])
        interpolated_images = eps * real_images + (1 - eps) * fake_images
        interpolated_images = tf.Variable(interpolated_images)
        return interpolated_images

    @tf.function
    def run_step(self, text_inputs, real_images, fake_images, interpolated_images):
        with tf.GradientTape() as tape:
            gradients = self._compute_gradients(interpolated_images, text_inputs)
            gradient_penalty = self._compute_gradient_penalty(gradients)
            real_logit = self._compute_critic_logit(real_images, text_inputs)
            fake_logit = self._compute_critic_logit(fake_images, text_inputs)
            w_dist = real_logit - fake_logit
            loss = cfg.WGAN_GP_LAMBDA * gradient_penalty
            if not self._gp_only:
                loss -= w_dist

        trainable_vars = [var for var in self._discriminator.variables if not var.name.startswith("text_rnn")]
        disc_gradients = tape.gradient(loss, trainable_vars)

        self._discriminator.optimizer.apply_gradients(zip(disc_gradients, trainable_vars))
        return [gradient_penalty, w_dist]
    
    def _compute_gradients(self, interpolated_images, text_inputs):
        with tf.GradientTape() as tape:
            inputs_dict = {"image": interpolated_images, **text_inputs}
            # Somehow feeding the dict does not work...
            inputs = [inputs_dict["chars"], inputs_dict["spec"], inputs_dict["image"]]
            interpolated_logits = self._discriminator(inputs, training=True)
        gradients = tape.gradient(interpolated_logits, [interpolated_images])
        return gradients[0]

    def _compute_gradient_penalty(self, gradients):
        gradients_sqr = Kb.square(gradients)
        gradients_sqr_sum = Kb.sum(gradients_sqr,
                                axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = Kb.sqrt(gradients_sqr_sum)
        gradient_penalty = Kb.square(1. - gradient_l2_norm)
        return Kb.mean(gradient_penalty)

    def _compute_critic_logit(self, images, text_inputs):
        inputs_dict = {"image": images, **text_inputs}
        # Somehow feeding the dict does not work...
        inputs = [inputs_dict["chars"], inputs_dict["spec"], inputs_dict["image"]]
        logits = self._discriminator(inputs, training=True)
        return Kb.mean(logits)


## LOSS FROM ORIGINAL GAN PAPER ##
    
binary_crossentropy = K.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def alternative_binary_crossentropy(y_true, y_pred):
    """ For the generator, we use log(1-D(G(z))) instead of -log(D(G(z))) """
    y_true = 1. - y_true
    return - binary_crossentropy(y_true, y_pred)


    


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    logits = tf.range(-5, 5, .1, dtype=tf.float32)
    logits = tf.expand_dims(logits, -1)
    y_pred = tf.math.tanh(logits)
    y_true = tf.ones_like(y_pred) * 0.5
    tce_1 = _tce(y_true, y_pred)
    plt.plot(y_pred, tce_1, "b+-")
    tce_2 = _tce_logit_abs(y_true, logits)
    plt.plot(y_pred, tce_2, "r--")

    plt.show()