from typing import Tuple, Union

import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.utils import resample
import scipy.stats as st
from sklearn.linear_model import LinearRegression, LassoCV

from pcs_inference.inference_methods.desparsified_lasso_inference import desparsified_lasso
from pcs_inference.inference_methods.lasso_mls import residual_mls


def confidence_interval(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray],
                        conf_level: float = 0.05, method: str = 'classic', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence intervals for the coefficients of a linear model.
    Args:
        X Union[pd.DataFrame, np.ndarray]: design matrix
        y Union[pd.DataFrame, np.ndarray]: response vector
        conf_level (float, optional): confidence level. Defaults to 0.05.
        method (str, optional): method to compute confidence intervals. Defaults to 'classic'.
        **kwargs:

    Returns:
        Tuple[np.ndarray, np.ndarray]: lower and upper bounds of the confidence intervals

    """
    if method == 'classic':
        return classic_confidence_interval(X, y, conf_level)
    elif method == 'bootstrap':
        return bootstrap_confidence_interval(X, y, conf_level)
    elif method == 'empirical_bootstrap':
        return empirical_bootstrap(X, y, conf_level)
    elif method == 'mls':
        return residual_mls(X, y)



    else:
        raise ValueError('The only methods available are classic and bootstrap')


def classic_confidence_interval(X, y, conf_level=0.05):
    # X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()
    cbs = res.conf_int(conf_level)
    return cbs[1:, 0], cbs[1:, 1]


def empirical_bootstrap(X, y, conf_level, model='linear', B=50):
    """
    Compute confidence intervals via empirical bootstrap.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    model : linear regression model

    B : int
        Number of bootstrap samples.

    """
    n = X.shape[0]

    if model == 'linear':
        reg = LinearRegression()
    elif model == 'lasso':
        reg = LassoCV()
    else:
        raise ValueError('The only regression method available is lasso and OLS')

    model_coefs = np.zeros((B, X.shape[1]))  # compute bootstrap residuals and refit model.
    reg.fit(X, y)  # fit model
    for i in range(B):
        bootstrapped_indices = resample([*range(n)])
        X_resampled = X[bootstrapped_indices, :]
        y_resampled = y[bootstrapped_indices]
        if model == 'linear':
            bootstrapped_model = LinearRegression()
        elif model == 'lasso':
            bootstrapped_model = LassoCV()
        else:
            raise ValueError('The only regression method available is lasso and OLS')
        bootstrapped_model.fit(X_resampled, y_resampled)
        model_coefs[i, :] = bootstrapped_model.coef_
    stds = np.std(model_coefs, axis=0)
    gaussian_quantile = st.norm.ppf(1 - conf_level / 2)
    cb_min = reg.coef_ - gaussian_quantile * stds
    cb_max = reg.coef_ + gaussian_quantile * stds
    return cb_min, cb_max


def bootstrap_confidence_interval(X, y, conf_level=0.05, num_iterations=50):
    model_coefs = np.zeros((num_iterations, X.shape[1]))
    for i in range(num_iterations):
        X_bootstrap, y_bootstrap = resample(X, y)
        m = LinearRegression().fit(X_bootstrap, y_bootstrap)
        model_coefs[i, :] = m.coef_
    confidence_intervals = []
    for k in range(X.shape[1]):
        confidence_intervals.append(
            st.norm.interval(confidence=1 - conf_level, loc=np.mean(model_coefs[:, k]), scale=st.tstd(model_coefs[:, k])))
    cb_min = [x[0] for x in confidence_intervals]
    cb_max = [x[1] for x in confidence_intervals]
    return cb_min, cb_max


if __name__ == '__main__':
    X, y = np.random.normal(size=(100, 10)), np.random.normal(size=(100,))
    cb_min, cb_max = confidence_interval(X, y, method='bootstrap')
    print(cb_min, cb_max)
    cb_min, cb_max = confidence_interval(X, y, method='empirical_bootstrap')
    print(cb_min, cb_max)
    cb_min, cb_max = confidence_interval(X, y, method='mls')
    print(cb_min, cb_max)
    cb_min, cb_max = confidence_interval(X, y, method='lasso')
    print(cb_min, cb_max)
    cb_min, cb_max = confidence_interval(X, y, method='classic')
    print(cb_min, cb_max)
