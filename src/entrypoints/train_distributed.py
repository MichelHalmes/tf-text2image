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
from data.encoders import get_encoders
from model import get_generator
import config
from evaluation import EvaluationLogger


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
    environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "worker": [f"localhost:1234{r}" for r in range(size)]
        },
        "task": {"type": "worker", "index": rank}
    })
    is_master = rank==0
    train(restore, is_master)


def train(restore, is_master=True):
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        encoders = get_encoders()
        dataset = get_dataset(encoders)
        train_data = dataset.batch(config.BATCH_SIZE)

        _, generator = get_generator(encoders)

        checkpoint_path = path.join(config.CHECKPOINT_DIR, "keras", "generator.ckpt")
        if restore:
            generator.load_weights(checkpoint_path)

    callbacks = []
    if is_master:
        generator.summary()
        stats_filename = datetime.now().strftime("%Y%m%d_%H%M") + ".csv"
        callbacks = [
            K.callbacks.CSVLogger(path.join(config.LOG_DIR, "stats", stats_filename)),
            # K.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True),
            EvaluationLogger(generator, dataset, encoders)
        ]
    initial_epoch = generator.optimizer.iterations.numpy() // config.STEPS_PER_EPOCH
    generator.fit(train_data, epochs=config.NUM_EPOCHS, initial_epoch=initial_epoch,
                steps_per_epoch=config.STEPS_PER_EPOCH,
                callbacks=callbacks)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s")

    train_distr()
