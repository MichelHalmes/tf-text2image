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


def add_noise(image):
    image += tf.random.truncated_normal(image.shape, stddev=.2)
    image = tf.clip_by_value(image, -1., 1.)
    return image

def shuffle_text(text_inputs_dict):
    shuffled = copy(text_inputs_dict)
    if random.random() < .7:
        shuffled["chars"] = tf.random.shuffle(shuffled["chars"])
        if random.random() < .1:
            shuffled["spec"] = tf.random.shuffle(shuffled["spec"])
    else:
        shuffled["spec"] = tf.random.shuffle(shuffled["spec"])
    return shuffled



def _get_train_on_batch_f(generator, discriminator, gan, accumulator):
    # @tf.function TODO: avoid eager and use summary writer
    gradient_penalizer = GradientPenalizer(discriminator)
    def _train_on_batch(text_inputs_dict, real_images, train_part=TRAIN_GD):
        fake_images = generator(text_inputs_dict, training=False)
        # real_images = add_noise(real_images)
        # fake_images = add_noise(fake_images)

        if _D in train_part:
            # Train discriminator on real images
            inputs_dict = {"image": real_images, **text_inputs_dict}
            labels = tf.ones(config.BATCH_SIZE)
            d_loss_real = discriminator.train_on_batch(inputs_dict, labels)
            accumulator.update(discriminator, d_loss_real)

            # Train discriminator on fake images
            inputs_dict = {"image": fake_images, **text_inputs_dict}
            labels = tf.zeros(config.BATCH_SIZE)
            d_loss_fake = discriminator.train_on_batch(inputs_dict, labels)
            accumulator.update(discriminator, d_loss_fake)

            # Train discriminator on real images
            inputs_dict = {"image": real_images, **text_inputs_dict}
            labels = tf.ones(config.BATCH_SIZE)
            d_loss_real = discriminator.train_on_batch(inputs_dict, labels)
            accumulator.update(discriminator, d_loss_real)

            # Train discriminator on images with wrong text
            shuffled_text_inputs_dict = shuffle_text(text_inputs_dict)
            inputs_dict = {"image": real_images, **shuffled_text_inputs_dict}
            labels = tf.zeros(config.BATCH_SIZE)
            d_loss_shuffle = discriminator.train_on_batch(inputs_dict, labels)
            accumulator.update(discriminator, d_loss_shuffle)

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
    dataset = get_dataset(encoders, difficulty=5)
    text_rnn, generator, discriminator, gan = get_models(encoders)

    checkpoint_path = path.join(config.CHECKPOINT_DIR, "keras", "text_rnn.ckpt")
    if restore:
        text_rnn.load_weights(checkpoint_path)

    logger = EvaluationLogger(generator, dataset, encoders)
    accumulator = MetricsAccumulator(path.join(config.LOG_DIR, "stats"))

    _train_on_batch_f = _get_train_on_batch_f(generator, discriminator, gan, accumulator)

    difficulty = -1
    dataset = get_dataset(encoders, difficulty)
    train_data = dataset.batch(config.BATCH_SIZE).take(config.STEPS_PER_EPOCH)
    for epoch in range(config.NUM_EPOCHS):
        # text_rnn.load_weights(checkpoint_path)  # TODO
        # if epoch > 500 and epoch % 20 == 0:
        #     difficulty += 0
        #     dataset = get_dataset(encoders, difficulty)
        #     train_data = dataset.batch(config.BATCH_SIZE).take(config.STEPS_PER_EPOCH)
        start_time = time.time()
        for b, (text_inputs_dict, images) in enumerate(train_data):
            print(f"{b} completed", end="\r")
            train_part = TRAIN_D if epoch < 7 else \
                        TRAIN_GD if b%2 == 0 else TRAIN_D
            _train_on_batch_f(text_inputs_dict, images, train_part)
        accumulator.accumulate(epoch)
        logger.on_epoch_end(epoch)
        logging.info("Done with epoch %s took %ss (difficulty=%s)", 
                        epoch, round(time.time()-start_time, 2), difficulty)


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
