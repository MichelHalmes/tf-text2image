import logging
import sys
from datetime import datetime
from os import path

import tensorflow.keras as K
import tensorflow.estimator as E
import tensorflow as tf
import pandas as pd

sys.path.append(path.abspath(path.join(__file__, "../../")))

from dataset import get_dataset, get_encoder
from model import get_model
import config
from evaluate import plot_samples


def train():
    encoder = get_encoder()
    ds = get_dataset(encoder)
    model = get_model(encoder)

    optimizer = K.optimizers.Adam()

    model.compile(loss=K.losses.mean_squared_error, 
                optimizer=optimizer,
                metrics=[K.losses.mean_absolute_error]
                )

    # latest = tf.train.latest_checkpoint(config.CHECKPOINT_DIR)
    # model.load_weights(latest)


    ### KERAS ###

    checkpoint_path = path.join(config.CHECKPOINT_DIR, "keras", "model.ckpt")
    stats_filename = datetime.now().strftime('%Y%m%d_%H%M') + ".csv"
    callbacks = [
        K.callbacks.TensorBoard("./data/logs/tf_boards"),
        K.callbacks.CSVLogger(f"./data/logs/stats/{stats_filename}.csv"),
        K.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    ]

    train_data = ds.batch(1)
    history = model.fit(train_data, epochs=2, validation_data=train_data, callbacks=callbacks)


    ### EVALUATE ###
    for texts, images in ds.batch(3).take(1):
        gen_images = model(texts)
        texts = [encoder.decode(text.numpy()) for text in texts]
        plot_samples(texts, gen_images, images)


if __name__ =="__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    train()
