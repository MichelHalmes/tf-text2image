import logging
import sys
from datetime import datetime
from os import path

import click
import tensorflow.keras as K
import tensorflow as tf

sys.path.append(path.abspath(path.join(__file__, "../../")))

from data.dataset import get_dataset
from data.encoders import get_encoders
from model import get_model
import config
from evaluation import EvaluationLogger
from utils import CustomSchedule, tanh_cross_entropy


@click.command()
@click.option("--restore/--no-restore", default=True, help="Reinititalize the model or restore previous checkpoint")
def train(restore):
    encoders = get_encoders()
    dataset = get_dataset(encoders)

    model = get_model(encoders)
    optimizer = K.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=tanh_cross_entropy,
                optimizer=optimizer,
                metrics=[K.losses.mean_squared_error, K.losses.mean_absolute_error]
    )

    checkpoint_path = path.join(config.CHECKPOINT_DIR, "keras", "model.ckpt")
    if restore:
        model.load_weights(checkpoint_path)
    model.summary()


    stats_filename = datetime.now().strftime('%Y%m%d_%H%M') + ".csv"
    callbacks = [
        # K.callbacks.TensorBoard(path.join(config.LOG_DIR, "tf_boards")),
        K.callbacks.CSVLogger(path.join(config.LOG_DIR, "stats", stats_filename)),
        K.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True),
        EvaluationLogger(model, dataset, encoders)
    ]

    initial_epoch = model.optimizer.iterations.numpy() // config.STEPS_PER_EPOCH
    train_data = dataset.batch(config.BATCH_SIZE).take(config.STEPS_PER_EPOCH)
    # val_data = dataset.batch(config.BATCH_SIZE).take(8)
    model.fit(train_data, epochs=config.NUM_EPOCHS, initial_epoch=initial_epoch,
                # validation_data=val_data, 
                callbacks=callbacks)



if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    train()
