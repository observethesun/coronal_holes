"""Supplementary utils."""
import numpy as np


def detect_edges(img):
    """Find edges in a binary image.

    Parameters
    ----------
    img : nbarray
        Binary image.

    Returns
    -------
    edge : ndarray
        Binary image of the same shape as img with saturated edges.
    """
    edge = np.zeros_like(img, dtype='int')

    d = np.diff(img, axis=0)
    ind = np.where(d == 1)
    edge[ind[0] + 1, ind[1]] = 1
    ind = np.where(d == -1)
    edge[ind[0], ind[1]] = 1

    d = np.diff(img, axis=1)
    ind = np.where(d == 1)
    edge[ind[0], ind[1] + 1] = 1
    ind = np.where(d == -1)
    edge[ind[0], ind[1]] = 1

    return edge
