
from os import path
import matplotlib.pyplot as plt

import config


def plot_samples(charss, specs, gen_images, orig_images):
    fig, axes = plt.subplots(nrows=len(charss), ncols=3)

    axes[0, 0].set_title("Input", weight="bold")
    axes[0, 1].set_title("Ground truth", weight="bold")
    axes[0, 2].set_title("Generated", weight="bold")

    for ax, chars, spec, gen_image, orig_image in zip(axes, charss, specs, gen_images, orig_images):
        ax[0].set_axis_off()
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
        txt = ax[0].text(.5, .5, chars+"\n"+spec, ha='center', va='center', wrap=True)
        txt._get_wrap_line_width = lambda : 150.
        ax[1].imshow(orig_image)
        ax[2].imshow(gen_image)

    fig_path = path.join(config.LOG_DIR, "evaluation.png")
    fig.savefig(fig_path)



