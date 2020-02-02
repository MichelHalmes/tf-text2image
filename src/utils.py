import logging

from tabulate import tabulate

import tensorflow.keras as K
import tensorflow as tf



def _product(iterable):
    p = 1
    for n in iterable:
        p *= n
    return p


def print_model_summary(model):
    """ Keras.summary() doesnt distinguish well be tween trainable and non-trainable parameters.
        Thise method is an alternative that does...
    """
    total_train_params, total_non_train_params = 0, 0
    table = [["Class", "Name", "Input-Names", "Output-Shape", "Train-params", "Non-Train-Params"]]
    for layer in model.layers:
        nb_train_params = sum(_product(weight.shape) for weight in layer.trainable_variables)
        total_train_params += nb_train_params
        nb_non_train_params = sum(_product(weight.shape) for weight in layer.non_trainable_variables)
        total_non_train_params += nb_non_train_params
        table.append((layer.__class__.__name__, layer.name, getattr(layer, "input_names", None), 
                                        layer.output_shape, nb_train_params, nb_non_train_params))
    table.append(("_________",)*6)
    table.append(("Model", model.name, model.input_names, model.output_shape, 
                                    total_train_params, total_non_train_params)) 

    print("\n", "#"*40, model.name.upper(), "#"*40)  
    print(tabulate(table, headers="firstrow", stralign="right"), "\n")
 


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

    temp_learning_rate_schedule = CustomSchedule(.001, 100)
    plt.plot(temp_learning_rate_schedule(tf.range(1000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")

    plt.show()

