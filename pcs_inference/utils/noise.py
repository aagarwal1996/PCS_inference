from sklearn.utils import resample

import numpy as np


def bootstrap(X, y):
    X_resample, y_resample = resample(X, y)
    return X_resample, y_resample


def add_normal_measurement_noise(y, sigma=0.1):
    y_perturbed = y + np.random.normal(0, sigma, size=y.shape)
    return y_perturbed


def add_laplace_measurement_noise(y, sigma=0.1):
    y_perturbed = y + np.random.laplace(0, sigma, size=y.shape)
    return y_perturbed


def add_X_noise(X, sigma=0.1):
    X_perturbed = X + np.random.normal(0, sigma, size=(X.shape[0], X.shape[1]))
    return X_perturbed


def remove_X_observations(X):
    return
