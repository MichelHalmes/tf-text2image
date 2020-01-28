import logging
import sys
from datetime import datetime
from os import path, environ
from collections import defaultdict
import csv
import random
import numpy.random

import click
import tensorflow.keras as K
import tensorflow as tf

sys.path.append(path.abspath(path.join(__file__, "../../")))

from data.dataset import get_dataset
from data.encoders import get_encoders
from model import get_models
import config
from evaluation import EvaluationLogger


import numpy as np

class MetricsAccumulator(object):

    _FIELDS = ["epoch", "discriminator_loss", "discriminator_binary_accuracy", 
            "gan_loss", "generator_loss", "generator_mean_squared_error", 
            "generator_mean_absolute_error"]

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

_G = "gen"
_D = "dis"
TRAIN_G = (_G,)
TRAIN_D = (_D,)
TRAIN_GD = (_G, _D)

def _get_train_on_batch_f(generator, discriminator, gan, accumulator):
    # @tf.function TODO: avoid eager and use summary writer
    def _train_on_batch(text_inputs_dict, images, train_part=TRAIN_GD):
        gen_images = generator(text_inputs_dict, training=False)

        if _D in train_part:
            # Train discriminator on real images
            inputs_dict = {"image": images, **text_inputs_dict}
            labels = tf.ones(config.BATCH_SIZE)  # TODO: one-sided label smoothing
            d_loss_real = discriminator.train_on_batch(inputs_dict, labels)

            # Train discriminator on fake images
            inputs_dict = {"image": gen_images, **text_inputs_dict}
            labels = tf.zeros(config.BATCH_SIZE)
            d_loss_fake = discriminator.train_on_batch(inputs_dict, labels)

            accumulator.update(discriminator, [(l1+l2)*.5 for l1, l2 in zip(d_loss_fake, d_loss_real)])
            # TODO: check if training fake & real at once is better
            # TODO: check if discrimitator training should be skipped occasionally

        # Train GAN
        if _G in train_part:
            labels = tf.ones(config.BATCH_SIZE)
            gan_loss = gan.train_on_batch(text_inputs_dict, labels)
            accumulator.update(gan, gan_loss)
        
        # Get generator metrics
        generator.reset_metrics()
        gen_loss = [f(images, gen_images).numpy() for f in generator.loss_functions+generator.metrics]
        accumulator.update(generator, gen_loss)
    return _train_on_batch



@click.command()
@click.option("--restore/--no-restore", default=True, help="Reinititalize the model or restore previous checkpoint")
def train(restore):
    encoders = get_encoders()
    dataset = get_dataset(encoders)
    text_rnn, generator, discriminator, gan = get_models(encoders)

    checkpoint_path = path.join(config.CHECKPOINT_DIR, "keras", "text_rnn.ckpt")
    if restore:
        text_rnn.load_weights(checkpoint_path)

    logger = EvaluationLogger(generator, dataset, encoders)
    accumulator = MetricsAccumulator(path.join(config.LOG_DIR, "stats"))

    _train_on_batch_f = _get_train_on_batch_f(generator, discriminator, gan, accumulator)
    train_data = dataset.batch(config.BATCH_SIZE).take(config.STEPS_PER_EPOCH)

    for epoch in range(config.NUM_EPOCHS):
        logging.info("Running epoch %s", epoch)
        for b, (text_inputs_dict, images) in enumerate(train_data):
            print(f"{b} completed", end="\r")
            train_part = TRAIN_D if epoch < 2 else \
                        TRAIN_GD  # if b%2 == 0 else TRAIN_G
            _train_on_batch_f(text_inputs_dict, images, train_part)
        accumulator.accumulate(epoch)
        logger.on_epoch_end(epoch)
            


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    random.seed(0)
    numpy.random.seed(0)
    tf.random.set_seed(0)
    environ["TF_DETERMINISTIC_OPS"] = "1"

    train()
