import numpy as np
from numpy.linalg import norm
from scipy.linalg import toeplitz, solve
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.model_selection import KFold


def reid(X, y, eps=1e-2, tol=1e-4, max_iter=int(1e4), n_jobs=1, seed=0):
    """Estimation of noise standard deviation using Reid procedure

    Parameters
    -----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    eps: float, optional (default=1e-2)
        Length of the cross-validation path.
        eps=1e-2 means that alpha_min / alpha_max = 1e-2.

    tol : float, optional (default=1e-4)
        The tolerance for the optimization: if the updates are smaller
        than `tol`, the optimization code checks the dual gap for optimality
        and continues until it is smaller than `tol`.

    max_iter : int, optional (default=1e4)
        The maximum number of iterations.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the cross validation.

    seed: int, optional (default=0)
        Seed passed in the KFold object which is used to cross-validate
        LassoCV. This seed controls the partitioning randomness.

    Returns
    -------
    sigma_hat : float
        Estimated noise standard deviation.

    beta_hat : array, shape (n_features,)
        Estimated parameter vector.

    References
    ----------
    .. [1] Reid, S., Tibshirani, R., & Friedman, J. (2016). A study of error
           variance estimation in lasso regression. Statistica Sinica, 35-67.
    """

    X = np.asarray(X)
    n_samples, n_features = X.shape

    if max_iter // 5 <= n_features:
        max_iter = n_features * 5
        print(f"'max_iter' has been increased to {max_iter}")

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    clf_lasso_cv = \
        LassoCV(eps=eps, fit_intercept=False,
                cv=cv, tol=tol, max_iter=max_iter, n_jobs=n_jobs) #normalize = False

    clf_lasso_cv.fit(X, y)
    beta_hat = clf_lasso_cv.coef_
    residual = clf_lasso_cv.predict(X) - y
    coef_max = np.max(np.abs(beta_hat))
    support = np.sum(np.abs(beta_hat) > tol * coef_max)

    # avoid dividing by 0
    support = min(support, n_samples - 1)

    sigma_hat = norm(residual) / np.sqrt(n_samples - support)

    return sigma_hat, beta_hat
