import logging
import sys
from os import environ, path

import tensorflow.keras as K
import tensorflow.estimator as E

sys.path.append(path.abspath(path.join(__file__, "../../")))

from data.dataset import get_dataset
from data.encoders import get_encoders
from model import get_generator
import config


def train():
    encoders = get_encoders()

    generator = get_generator(encoders)

    checkpoint_dir = path.join(config.CHECKPOINT_DIR, "tf")
    estimator = K.estimator.model_to_estimator(keras_model=generator,
                                        model_dir=checkpoint_dir,
                                        checkpoint_format="saver")  # TODO: use 'checkpoint' once object-based checkpoints supported

    def input_fn():
        dataset = get_dataset(encoders)
        return dataset.batch(config.BATCH_SIZE)

    train_spec = E.TrainSpec(input_fn=input_fn)
    eval_spec = E.EvalSpec(
        input_fn=input_fn,
        hooks=[E.CheckpointSaverHook(checkpoint_dir, save_steps=1000)])

    E.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    train()
