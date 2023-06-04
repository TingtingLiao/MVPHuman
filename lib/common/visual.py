import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def image_grid(
        images,
        rows=None,
        cols=None,
        show=False,
        fill: bool = False,
        show_axes: bool = False,
        rgb: bool = True,
        save_path: str = None,
        dpi: int = 300
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.
        save_path: image save path
    Returns:
        None
    """
    if len(images) == 0:
        return
    H, W, _ = images[0].shape
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0., "hspace": 0.}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(10, 10. * rows / cols * H / W))
    bleed = 0.
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed), wspace=0., hspace=0.)
    plt.margins(0, 0)

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()

    if save_path is not None:
        fig.savefig(save_path, pad_inches=0.0, dpi=dpi)
    if show:
        plt.show()
    plt.close()


