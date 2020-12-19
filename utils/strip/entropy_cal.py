import scipy
import numpy as np
import tqdm.notebook as tq

from utils.strip.super_impose import superImpose


def entropyCal(background, clean_set, model, overlay_weight=0.5, back_weight=0.9):
    """
    Used to calculate mean entropy of `background` image
    superimpose with each image in `clean_set`.
    Usage:
    >>> H = entropyCal(x[0], 10, x_valid, model, 0.8, 0.8)
    Arguments:
        background: origin input (the image to be tested)
        clean_set: clean images to superimpose
        model: model used to predict
        overlay_weight: weight of overlay_img.
        back_weight: weight of origin_img.
    Returns:
        H: mean entropy.
    """
    x_perturb = []  # list of perturbed image
    index_overlay = np.random.randint(0, clean_set.shape[0], size=10)
    H = 0
    for i in range(10):
        x_perturb.append( superImpose(clean_set[index_overlay[i]], background, overlay_weight, back_weight) )
        predictions = model(np.expand_dims(x_perturb[i], axis=0)).numpy()
        Hn = 0.0
        for p in predictions[0]:
            if p==0.0:  # log2(0) will cause nan value in H
                continue
            Hn += -p * np.log2(p)
        H += Hn
    H /= 10
    return H


def getEntropyList(x_test, x_valid, model, overlay_weight=0.5, back_weight=0.9):
    """
    Used to compute list of entropy of `x_test` superimpose with images in `x_valid`.
    Usage:
    >>> entropy = getEntropyList(n_test, n_sample, x_test_clean, x_valid, bd_model, overlay_weight, back_weight)
    Arguments:
        x_test: image to be tested
        x_valid: clean validation data
        model: model to be tested
        overlay_weight: weight of overlay_img.
        back_weight: weight of origin_img.
    Returns:
        entropy: list of entropy.
    """
    entropy = []
    n_test = len(x_test)
    for j in tq.tqdm(range(n_test)):
        x_background = x_test[j]
        entropy.append(entropyCal(x_background, x_valid, model, overlay_weight, back_weight))
    return entropy


def computeThreshold(entropy_benigh, frr=0.07):
    """
    Used to compute threshold.
    Test image with entropy less than threshold is considered to be backdoor image,
    otherwise benigh(clean) image.
    Usage:
    >>> threshold = computeThreshold(entropy_benigh, 0.05)
    Arguments:
        entropy_benigh: list of entropy of clean input superimpose with clean input
        frr: preset False Reject Rate in entropy of clean img
    Returns:
        threshold: threshold computed.
    """
    (mu, sigma) = scipy.stats.norm.fit(entropy_benigh)
    print(f"Clean image: Mean={mu}, Var={sigma}")

    threshold = scipy.stats.norm.ppf(frr, loc=mu, scale=sigma)
    print(f"Computed threshold is {threshold}")
    return threshold
