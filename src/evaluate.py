
from os import path
import matplotlib.pyplot as plt

import config


def plot_samples(texts, gen_images, orig_images):

    fig, axes = plt.subplots(nrows=len(texts), ncols=3)

    for ax, text, gen_image, orig_image in zip(axes, texts, gen_images, orig_images):
        ax[0].set_axis_off()
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
        ax[0].text(.5, .5, text)
        ax[1].imshow(orig_image)
        ax[2].imshow(gen_image)

    fig_path = path.join(config.LOG_DIR, "evaluation.png")
    fig.savefig(fig_path)



