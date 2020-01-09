import logging
import sys
from os import environ, path

import tensorflow.keras as K
import tensorflow.estimator as E

sys.path.append(path.abspath(path.join(__file__, "../../")))

from dataset import get_dataset, get_encoder
from model import get_model
import config

def train():
    encoder = get_encoder()
    model = get_model(encoder)

    optimizer = K.optimizers.Adam()
    model.compile(loss=K.losses.mean_squared_error, 
                optimizer=optimizer,
                metrics=[K.losses.mean_absolute_error]
    )

    checkpoint_dir = path.join(config.CHECKPOINT_DIR, "tf")
    estimator = K.estimator.model_to_estimator(keras_model=model, 
                                        model_dir=checkpoint_dir, 
                                        checkpoint_format='saver')  # TODO: use 'checkpoint once object-based checkpoints supported

    def input_fn():
        return get_dataset(encoder).batch(4)

    train_spec = E.TrainSpec(input_fn=input_fn, max_steps=1000)
    eval_spec = E.EvalSpec(input_fn=input_fn)

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
