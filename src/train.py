import logging
import sys

from dataset import get_dataset, get_encoder

def train():
    encoder = get_encoder()
    ds = get_dataset(encoder)

    for text, image in ds.take(1):
        print(text)
        # print(encoder.encode(text.numpy()))
        print(image)


if __name__ =="__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    train()
