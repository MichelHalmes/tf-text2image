import tensorflow.keras.backend as Kb
import tensorflow.keras as K
import tensorflow as tf

import numpy as np

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
        y_true: (fake, real)=(0, 1)
            should be mapped to the multipliers for f: (1, -1)
    """
    multiplier = y_true*-2. + 1.
    return Kb.mean(multiplier * y_pred, axis=-1)


def clip_weights(network, clip_value):
    """ Used in combination with Wasserstein loss """
    for layer in network.layers:
        weights = layer.get_weights()
        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
        layer.set_weights(weights)

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