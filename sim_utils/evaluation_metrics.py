import numpy as np
import pandas as pd


def get_length(cb_min,cb_max):
    return np.abs(cb_max - cb_min)

def check_if_covers(beta,cb_min,cb_max):
    indicator = [0]*len(beta)
    for i in range(len(indicator)):
        if (cb_min[i] <= beta[i]) and (beta[i] <= cb_max[i]):
            indicator[i] = 1
        else:
            continue
    return indicator


