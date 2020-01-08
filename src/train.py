import logging
import sys

import tensorflow.keras as K
import tensorflow.estimator as E

from dataset import get_dataset, get_encoder
from model import get_model

def train():
    encoder = get_encoder()
    ds = get_dataset(encoder)
    model = get_model(encoder)
    # model = K.models.Model(model)

    optimizer = K.optimizers.Adam()

    model.compile(loss=K.losses.mean_squared_error, 
                optimizer=optimizer,
                # metrics=[K.losses.mean_absolute_error]
                )
    # ### MANUAL ###
    for text, image in ds.batch(1).take(1):
        # print(text.numpy())
        # print(encoder.decode(text.numpy()[0]))
        gen_image = model(text)
        # print(gen_image)

    ### KERAS ###
    train_data = ds.batch(2) #, [(2,), (128, 128, 3)]) # TODO: padded batch
    history = model.fit(train_data,
                    epochs=10,
                    # validation_data=None,validation_steps=30
                    )


    ### ESTIMATOR ###
    # model_dir = "./data/model"
    # estimator = K.estimator.model_to_estimator(keras_model=model, 
    #                                     model_dir=model_dir, 
    #                                     checkpoint_format='saver')  # TODO: use 'checkpoint once object-based checkpoints supported

    # def input_fn():
    #     return get_dataset(encoder).batch(4)

    # train_spec = E.TrainSpec(input_fn=input_fn, max_steps=1000)
    # eval_spec = E.EvalSpec(input_fn=input_fn)

    # E.train_and_evaluate(
    #     estimator,
    #     train_spec,
    #     eval_spec
    # )


if __name__ =="__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    train()
