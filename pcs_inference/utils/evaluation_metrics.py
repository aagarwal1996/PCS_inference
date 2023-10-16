import numpy as np


def get_length(cb_min, cb_max):
    # make sure cb_min and cb_max are numpy arrays
    cb_min = np.array(cb_min)
    cb_max = np.array(cb_max)
    return np.abs(cb_max - cb_min)


def check_if_covers(beta, cb_min, cb_max):
    indicator = [0] * len(beta)
    for i in range(len(indicator)):
        if (cb_min[i] <= beta[i]) and (beta[i] <= cb_max[i]):
            indicator[i] = 1
        else:
            continue
    return indicator
