from collections import defaultdict
from os import path
import csv
from datetime import datetime

import tensorflow.keras as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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
    plt.close(fig)


class EvaluationLogger(K.callbacks.Callback):

    def __init__(self, generator, dataset, encoders):
        self._generator = generator
        self._data_iter = iter(dataset.batch(3))
        self._encoders = encoders
        

    def on_epoch_end(self, epoch, logs=None):   
        input_dict, images = next(self._data_iter)
        gen_images = self._generator(input_dict)

        gen_images = self._encoders.image.decode(gen_images)
        images = self._encoders.image.decode(images)
        gen_images = tf.clip_by_value(gen_images, 0., 1.)

        charss = [self._encoders.chars.decode(chars.numpy()) \
                        for chars in input_dict["chars"]]
        specs = [self._encoders.spec.decode(spec.numpy()) \
                        for spec in input_dict["spec"]]
        plot_samples(charss, specs, gen_images, images)


class MetricsAccumulator(object):

    _FIELDS = ["epoch", "discriminator_loss", "discriminator_binary_accuracy", 
            "gan_loss", "generator_loss", "generator_mean_squared_error", 
            "generator_mean_absolute_error", "gradient_penalizer_rms"]

    def __init__(self, log_dir):
        self._metric_values = defaultdict(list)
        self._metric_names = {}
        stats_filename = datetime.now().strftime('%Y%m%d_%H%M') + ".csv"
        stats_path = path.join(log_dir, stats_filename)
        self._csv_file = open(stats_path, "w")
        self._csv_writer =  None

    def _init_writer(self, field_names):
        writer = csv.DictWriter(self._csv_file, fieldnames=field_names)
        writer.writeheader()
        return writer

    def update(self, model, metrics):
        if not isinstance(metrics, list):
            metrics = [metrics]

        key = model.name
        self._metric_values[key].append(metrics)
        self._metric_names[key] = model.metrics_names

    def accumulate(self, epoch):
        metrics_dict = {"epoch": epoch}
        for key, values in self._metric_values.items():
            values = np.array(values)
            values = np.mean(values, axis=0).tolist()
            bing = {f"{key}_{name}": value for name, value in zip(self._metric_names[key], values)}
            metrics_dict.update(bing)
        if self._csv_writer is None:
            # self._csv_writer = self._init_writer(metrics_dict.keys())
            self._csv_writer = self._init_writer(self._FIELDS)
        self._csv_writer.writerow(metrics_dict)
        self._csv_file.flush()
        self._metric_values = defaultdict(list)


