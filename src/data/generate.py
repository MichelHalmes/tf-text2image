import random
from contextlib import suppress

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import config

COLORS = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]


def _sample_chars(difficulty=-1):
    alphabeth = config.CHARS_ALPHABETH[:difficulty] if difficulty > 1 else config.CHARS_ALPHABETH
    return "".join(random.choice(alphabeth) for _ in range(config.CHARS_LENGTH))

def _sample_spec_dict():
    backgroundcolor = random.choice(COLORS)
    color = random.choice(COLORS)
    color = random.choice(COLORS) if color == backgroundcolor else color  # Reduce the probability of having just a colored square
    return {
        "backgroundcolor": backgroundcolor,
        "color": color
    }

def _spec_dict_as_text(spec_dict):
    return ", ".join(f"{key} {value}" for key, value in spec_dict.items())

def _plot_sample(chars, spec_dict):
    fig, ax = plt.subplots(1, frameon=False)
    ax.set_position([0., 0., 1., 1.])
    ax.set_axis_off()
    ax.text(0.5, 0.5, chars, 
            horizontalalignment="center", verticalalignment="center", 
            fontsize=300, **spec_dict)

    return fig

def _fig_to_numpy(fig):
    fig.canvas.draw()
    image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    H, W = fig.canvas.get_width_height()
    image_np = image_np.reshape((W, H, 3))
    plt.close(fig)

    return image_np

def generate_sample(difficulty=-1):
    chars = _sample_chars(difficulty)
    spec_dict = _sample_spec_dict()
    spec_ = _spec_dict_as_text(spec_dict)
    fig = _plot_sample(chars, spec_dict)
    image_np = _fig_to_numpy(fig)

    return chars, spec_, image_np


