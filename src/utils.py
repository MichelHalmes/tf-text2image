import logging

import tensorflow.keras as K
import tensorflow.keras.backend as Kb
import tensorflow as tf


def tanh_cross_entropy(y_true, y_pred):
    # See README for details 
    use_logits, logits = _get_tanh_logits(y_pred)
    if use_logits:
        # We use the numerically stable method from logits
        return _tce_logit_abs(y_true, logits)
    else:
        logging.warn("Caution using tce-loss without logits")
        return _tce(y_true, y_pred)

def _get_tanh_logits(y_pred):
    if y_pred.op.type == "Tanh":
        assert len(y_pred.op.inputs) == 1
        logits = y_pred.op.inputs[0]
        return True, logits
    elif y_pred.op.type == "Identity":
        assert len(y_pred.op.inputs) == 1
        y_pred = y_pred.op.inputs[0]
        return _get_tanh_logits(y_pred)
    else: 
        print(y_pred.op.inputs[0].op.type)
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
 


class CustomSchedule(K.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_steps):
        super().__init__()
        self.max_lr = tf.cast(max_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.max_lr * tf.math.minimum(arg1, arg2) / tf.math.rsqrt(self.warmup_steps)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure(0)
    temp_learning_rate_schedule = CustomSchedule(.001, 100)
    plt.plot(temp_learning_rate_schedule(tf.range(1000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")

    plt.figure(1)
    logits = tf.range(-5, 5, .1, dtype=tf.float32)
    logits = tf.expand_dims(logits, -1)
    y_pred = tf.math.tanh(logits)
    y_true = tf.ones_like(y_pred) * 0.5
    tce_1 = _tce(y_true, y_pred)
    plt.plot(y_pred, tce_1, "b+-")
    tce_2 = _tce_logit_abs(y_true, logits)
    plt.plot(y_pred, tce_2, "r--")

    plt.show()




