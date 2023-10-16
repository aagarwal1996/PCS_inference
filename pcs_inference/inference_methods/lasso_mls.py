import numpy as np

from sklearn.linear_model import LassoCV, LinearRegression

from sklearn.utils import resample

from ..utils.dgp import linear_model


def mLs(X, y, tau="inverse_sample_size", cv=3):
    """Modified Least Squares with confidence intervals

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].

    Returns
    -------
    beta_hat : array, shape (n_features,)
        Estimated parameter vector.


    References
    ----------
    .. [1] Liu, & Yu. Asympptotic properties of Lasso+mLS and Lasso+Ridge in sparse high-dimensional linear regression
    """
    X = X - np.mean(X, axis=0)  # remove intercept
    y = y - np.mean(y)

    if tau == "inverse_sample_size":
        tau = 1.0 / X.shape[0]
    Lasso_reg = LassoCV(cv=cv).fit(X, y)
    selected_coefs = []
    for j in range(len(Lasso_reg.coef_)):
        if Lasso_reg.coef_[j] == 0:
            continue
        else:
            selected_coefs.append(j)
    X_reduced = X[:, selected_coefs] / np.sqrt(X.shape[0])
    U, S, Vh = np.linalg.svd(X_reduced, full_matrices=True)
    S_inverse = np.zeros(S.shape)
    for i, s in enumerate(S_inverse):
        if S[i] > tau:
            S_inverse[i] = 1.0 / S[i]
        else:
            continue
    smat = np.zeros((X_reduced.shape[1], X.shape[0]))
    smat[:X_reduced.shape[1], : X_reduced.shape[1]] = np.diag(S_inverse)
    beta_mls = np.dot(U.transpose(), y)
    beta_mls = np.dot(smat, beta_mls)
    beta_mls = (1 / np.sqrt(X.shape[0])) * np.dot(Vh.transpose(), beta_mls)
    coef_mls = np.zeros(X.shape[1])
    for idx, coef in enumerate(selected_coefs):
        coef_mls[coef] = beta_mls[idx]
    return coef_mls


def residual_mls(X, y, B=5, tau="inverse_sample_size", cv=3):
    n = X.shape[0]
    cb_min = [0] * X.shape[1]
    cb_max = [0] * X.shape[1]

    coef_ = mLs(X, y, tau, cv)
    preds = np.dot(X, coef_)
    residuals = y - preds  # compute residuals
    residuals = residuals - np.mean(residuals)  # center residuals
    model_coefs = np.zeros((B, X.shape[1]))  # store coefficients across bootstrap samples
    for i in range(B):  # compute bootstrap residuals and refit model.
        bootstrapped_indices = resample([*range(n)])
        X_resampled = X[bootstrapped_indices, :]
        y_resampled = preds[bootstrapped_indices] + residuals[bootstrapped_indices]
        model_coefs[i, :] = mLs(X_resampled, y_resampled, tau, cv)
    stds = np.std(model_coefs, axis=0)
    cb_min = coef_ - 1.96 * stds
    cb_max = coef_ + 1.96 * stds
    return cb_min, cb_max


if __name__ == '__main__':
    n_samples, n_features = 100, 10
    support_size = 3
    sigma = 0.01
    rho = 0.0

    X = np.random.normal(size=(n_samples, n_features))
    y, support, beta_s = linear_model(X=X, sigma=sigma, s=support_size, beta=1.0, return_support=True)
    cb_min, cb_max = residual_mls(X, y)
    print(cb_min, cb_max)
    # beta_hat, cb_min, cb_max = desparsified_lasso(X, y)
