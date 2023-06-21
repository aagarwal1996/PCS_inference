import numpy as np
import pandas as pd


def get_length(confidence_interval):
    return np.abs(confidence_interval[1] - confidence_interval[0])

