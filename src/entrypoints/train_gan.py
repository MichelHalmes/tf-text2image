import logging
import sys
from os import path, environ
import time
from copy import copy

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
from evaluation import EvaluationLogger, MetricsAccumulator
from losses import clip_weights, GradientPenalizer


_G = "gen"
_D = "dis"
TRAIN_G = (_G,)
TRAIN_D = (_D,)
TRAIN_GD = (_G, _D)


def _add_noise(image):
    image += tf.random.truncated_normal(image.shape, stddev=.2)
    image = tf.clip_by_value(image, -1., 1.)
    return image


def _shuffle_text(text_inputs_dict):
    shuffled = copy(text_inputs_dict)
    if random.random() < .75:
        shuffled["chars"] = tf.random.shuffle(shuffled["chars"])
        if random.random() < .2:
            shuffled["spec"] = tf.random.shuffle(shuffled["spec"])
    else:
        shuffled["spec"] = tf.random.shuffle(shuffled["spec"])
    return shuffled


def _get_train_discriminator_f(discriminator, accumulator):
    def train_discriminator(images, text_inputs_dict, is_real):
        inputs_dict = {"image": images, **text_inputs_dict}
        labels = tf.ones(config.BATCH_SIZE) if is_real \
                    else tf.zeros(config.BATCH_SIZE)
        d_loss = discriminator.train_on_batch(inputs_dict, labels)
        accumulator.update(discriminator, d_loss)
    return train_discriminator


def _get_train_on_batch_f(generator, discriminator, gan, accumulator):
    gradient_penalizer = GradientPenalizer(discriminator, gp_only=False)
    train_discriminator = _get_train_discriminator_f(discriminator, accumulator)

    def _train_on_batch(text_inputs_dict, real_images, train_part=TRAIN_GD):
        fake_images = generator(text_inputs_dict, training=False)

        if _D in train_part:
            if config.USE_WGAN_GP:
                # Minimize gradient penalty
                gp_loss = gradient_penalizer.run_on_batch(text_inputs_dict, real_images, fake_images)
                accumulator.update(gradient_penalizer, gp_loss)
                shuffled_images = tf.random.shuffle(real_images)
                gp_loss = gradient_penalizer.run_on_batch(text_inputs_dict, real_images, shuffled_images)
                accumulator.update(gradient_penalizer, gp_loss)
            else:
                # Train discriminator
                train_discriminator(fake_images, text_inputs_dict, is_real=False)
                train_discriminator(real_images, text_inputs_dict, is_real=True)
                shuffled_text_inputs_dict = _shuffle_text(text_inputs_dict)
                train_discriminator(real_images, shuffled_text_inputs_dict, is_real=False)
                train_discriminator(real_images, text_inputs_dict, is_real=True)

        # Train GAN
        if _G in train_part:
            labels = tf.ones(config.BATCH_SIZE)
            gan_loss = gan.train_on_batch(text_inputs_dict, labels)
            accumulator.update(gan, gan_loss)

        # Get generator metrics
        generator.reset_metrics()
        gen_loss = [f(real_images, fake_images).numpy() for f in generator.loss_functions+generator.metrics]
        accumulator.update(generator, gen_loss)
    return _train_on_batch


@click.command()
@click.option("--restore/--no-restore", default=True, help="Reinititalize the model or restore previous checkpoint")
def train(restore):
    encoders = get_encoders()
    dataset = get_dataset(encoders, difficulty=10)
    text_rnn, generator, discriminator, gan = get_models(encoders)

    checkpoint_path = path.join(config.CHECKPOINT_DIR, "keras", "text_rnn.ckpt")
    if restore:
        text_rnn.load_weights(checkpoint_path)

    logger = EvaluationLogger(generator, dataset, encoders)
    accumulator = MetricsAccumulator(path.join(config.LOG_DIR, "stats"))

    _train_on_batch_f = _get_train_on_batch_f(generator, discriminator, gan, accumulator)

    difficulty = 10
    dataset = get_dataset(encoders, difficulty)
    train_data = dataset.batch(config.BATCH_SIZE).take(config.STEPS_PER_EPOCH)
    for epoch in range(config.NUM_EPOCHS):
        # if epoch >= 500 and epoch % 10==0:
        #     difficulty += 1
        #     dataset = get_dataset(encoders, difficulty)
        #     train_data = dataset.batch(config.BATCH_SIZE).take(config.STEPS_PER_EPOCH)
        start_time = time.time()
        discr_only_steps = 0  # if epoch < 500 else 1
        for b, (text_inputs_dict, images) in enumerate(train_data):
            print(f"{b} completed", end="\r")
            train_part = TRAIN_D if epoch < 5 else \
                        TRAIN_GD if b%(discr_only_steps+1) == 0 else TRAIN_D
            _train_on_batch_f(text_inputs_dict, images, train_part)
        accumulator.accumulate(epoch)
        logger.on_epoch_end(epoch)
        logging.info("Done with epoch %s took %ss (difficulty=%s; discr_only_steps=%s)",
                        epoch, round(time.time() - start_time, 2), difficulty, discr_only_steps)


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
