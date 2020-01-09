import random

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import config

COLORS = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]


def _sample_chars():
    return "".join(random.choice(config.CHARS_ALPHABETH) for _ in range(config.CHARS_LENGTH))


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
            fontsize=150, **spec_dict)
    fig.set_facecolor(spec_dict["backgroundcolor"])
    return fig

def _fig_to_numpy(fig):
    fig.canvas.draw()
    image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    H, W = fig.canvas.get_width_height()
    image_np = image_np.reshape((W, H, 3))

    return image_np

def generate_sample():
    chars = _sample_chars()
    spec_dict = _sample_spec_dict()
    spec_ = _spec_dict_as_text(spec_dict)
    fig = _plot_sample(chars, spec_dict)
    image_np = _fig_to_numpy(fig)
    plt.close(fig)
    return chars, spec_, image_np


