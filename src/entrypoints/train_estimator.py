import logging
import sys
from os import environ, path

import tensorflow.keras as K
import tensorflow.estimator as E

sys.path.append(path.abspath(path.join(__file__, "../../")))

from data.dataset import get_dataset
from data.encoders import get_chars_encoder, get_spec_encoder
from model import get_model
import config

def train():
    chars_encoder = get_chars_encoder()
    spec_encoder = get_spec_encoder()

    model = get_model(chars_encoder, spec_encoder)
    optimizer = K.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=K.losses.mean_squared_error, 
                optimizer=optimizer,
                metrics=[K.losses.mean_absolute_error]
    )

    checkpoint_dir = path.join(config.CHECKPOINT_DIR, "tf")
    estimator = K.estimator.model_to_estimator(keras_model=model, 
                                        model_dir=checkpoint_dir, 
                                        checkpoint_format='saver')  # TODO: use 'checkpoint' once object-based checkpoints supported

    def input_fn():
        dataset = get_dataset(chars_encoder, spec_encoder)
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


if __name__ =="__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    train()
