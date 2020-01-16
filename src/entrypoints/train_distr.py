import logging
import sys
from datetime import datetime
from os import path, environ
from multiprocessing import Process
import json

import click
import tensorflow.keras as K
import tensorflow as tf

sys.path.append(path.abspath(path.join(__file__, "../../")))

from data.dataset import get_dataset
from data.encoders import get_chars_encoder, get_spec_encoder
from model import get_model
import config
from evaluation import EvaluationLogger
from utils import CustomSchedule


@click.command()
@click.option("--restore/--no-restore", default=True, help="Reinititalize the model or restore previous checkpoint")
def train_distr(restore):
    SIZE = 2
    processes = []
    for rank in range(SIZE):
        p = Process(target=_init_process, args=(restore, rank, SIZE))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def _init_process(restore, rank, size):
    environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': [f"localhost:1234{r}" for r in range(size)]
        },
        'task': {'type': 'worker', 'index': rank}
    })
    is_master = rank==0
    train(restore, is_master)


def train(restore, is_master=True):
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        chars_encoder = get_chars_encoder()
        spec_encoder = get_spec_encoder()
        dataset = get_dataset(chars_encoder, spec_encoder)

        model = get_model(chars_encoder, spec_encoder)
        optimizer = K.optimizers.Adam(learning_rate=0.001)
        model.compile(loss=K.losses.mean_squared_error, 
                    optimizer=optimizer,
                    metrics=[K.losses.mean_absolute_error]
        )
        checkpoint_path = path.join(config.CHECKPOINT_DIR, "keras", "model.ckpt")
        if restore:
            model.load_weights(checkpoint_path)


        initial_epoch = model.optimizer.iterations.numpy() // config.STEPS_PER_EPOCH
        train_data = dataset.batch(config.BATCH_SIZE)

    callbacks = []
    if is_master:
        model.summary()
        stats_filename = datetime.now().strftime('%Y%m%d_%H%M') + ".csv"
        callbacks = [
            K.callbacks.CSVLogger(path.join(config.LOG_DIR, "stats", stats_filename)),
            # K.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True),
            EvaluationLogger(model, dataset, chars_encoder, spec_encoder)
        ]

    model.fit(train_data, epochs=config.NUM_EPOCHS, initial_epoch=initial_epoch,
                steps_per_epoch=config.STEPS_PER_EPOCH,
                callbacks=callbacks)



if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s")

    train_distr()
