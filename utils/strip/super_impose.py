import numpy as np


def superImpose(overlay_img, origin_img, overlay_weight, back_weight):
    """
    Used to superimpose two images of format `numpy.ndarray`.
    The shape of the input should be exactly the same.
    Usage:
    >>> imgCombine = superImpose(x[0], x[1], 0.8, 0.8)
    Arguments:
        overlay_img: image used to perturb.
        origin_img: image to be tested.
        overlay_weight: weight of overlay_img.
        back_weight: weight of origin_img.
    Returns:
        ret: the linearly combined image.
    """
    ret = overlay_weight * overlay_img + back_weight * origin_img
    ret = np.clip(ret, 0.0, 1.0)
    return ret
