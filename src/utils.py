import logging

from tabulate import tabulate

import tensorflow.keras as K
import tensorflow as tf

import config


def _product(iterable):
    p = 1
    for n in iterable:
        p *= n
    return p


def _get_input_names(layer):
    inputs = layer.input
    if not isinstance(inputs, list):
        inputs = [inputs]
    return [get_short_name(node) for node in inputs]
    

def get_short_name(node):
    name = node.name
    if name.endswith(":0"):
        name = name[:-2]
    if name.endswith("/Identity"):
        name = name[:-9]
    return name


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
        input_names = _get_input_names(layer)
        output_shape = layer.output.shape
        table.append((layer.__class__.__name__, layer.name, input_names, 
                                        output_shape, nb_train_params, nb_non_train_params))
    table.append(("_________",)*6)
    table.append(("Model", model.name, model.input_names, model.output_shape, 
                                    total_train_params, total_non_train_params)) 

    print("\n", "#"*40, model.name.upper(), "#"*40)  
    print(tabulate(table, headers="firstrow", stralign="right"), "\n")
 


class CustomLrSchedule(K.optimizers.schedules.LearningRateSchedule):
    def __init__(self, ):
        super().__init__()
        self._max_lr = tf.cast(config.DIS_LR, tf.float32)
        self._half_steps = tf.cast(config.DIS_LR_DECAY_EPOCH*config.STEPS_PER_EPOCH, tf.float32)
        
    def __call__(self, step):
        self._curr_lr = self._max_lr * tf.math.pow(config.DIS_LR_DECAY, tf.math.floor((1+step)/self._half_steps))
        return self._curr_lr


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    temp_learning_rate_schedule = CustomLrSchedule()
    plt.plot(temp_learning_rate_schedule(tf.range(100000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")

    plt.show()

