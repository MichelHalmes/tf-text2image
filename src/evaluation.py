
from os import path

import tensorflow.keras as K
import tensorflow as tf
import matplotlib.pyplot as plt

import config



def plot_samples(charss, specs, gen_images, orig_images):
    fig, axes = plt.subplots(nrows=len(charss), ncols=3)

    axes[0, 0].set_title("Input", weight="bold")
    axes[0, 1].set_title("Ground truth", weight="bold")
    axes[0, 2].set_title("Generated", weight="bold")

    for ax, chars, spec, gen_image, orig_image in zip(axes, charss, specs, gen_images, orig_images):
        ax[0].set_axis_off()
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
        txt = ax[0].text(.5, .5, chars+"\n"+spec, 
                        ha='center', va='center', wrap=True)
        txt._get_wrap_line_width = lambda : 150.
        ax[1].imshow(orig_image)
        ax[2].imshow(gen_image)

    fig_path = path.join(config.LOG_DIR, "evaluation.png")
    fig.savefig(fig_path)


class EvaluationLogger(K.callbacks.Callback):

    def __init__(self, model, dataset, chars_encoder, spec_encoder):
        self._model = model
        self._data_iter = iter(dataset.batch(3))
        self._chars_encoder = chars_encoder
        self._spec_encoder = spec_encoder
        

    def on_epoch_end(self, epoch, logs=None):   
        input_dict, images = next(self._data_iter)
        gen_images = self._model(input_dict)
        # gen_images = tf.clip_by_value(gen_images, 0., 1.)

        charss = [self._chars_encoder.decode(chars.numpy()) \
                        for chars in input_dict["chars"]]
        specs = [self._spec_encoder.decode(spec.numpy()) \
                        for spec in input_dict["spec"]]
        plot_samples(charss, specs, gen_images, images)



