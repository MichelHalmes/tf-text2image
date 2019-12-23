import random

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

import config


def iter_data():
    while True:
        yield _generate_sample()

def _sample_text():
    text = "".join(random.choice(config.TEXT_ALPHABETH) for _ in range(config.TEXT_LENGTH))
    return text

def _plot_sample(text):
    fig, ax = plt.subplots(1, frameon=False)
    ax.set_position([0., 0., 1., 1.])
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', fontsize=200)
    # plt.show()
    # raise
    return fig

def _fig_to_numpy(fig):
    fig.canvas.draw()
    image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    H, W = fig.canvas.get_width_height()
    image_np = image_np.reshape((W, H, 3))
    return image_np

def _generate_sample():
    text = _sample_text()
    fig = _plot_sample(text)
    image_np = _fig_to_numpy(fig)
    return text, image_np





