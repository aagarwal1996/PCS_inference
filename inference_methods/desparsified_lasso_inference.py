import numpy as np
from numpy.linalg import multi_dot
from scipy import stats
from scipy.linalg import inv
from joblib import Parallel, delayed
from sklearn.utils.validation import check_memory
from sklearn.linear_model import Lasso

from noise_estimation import reid
from numpy.testing import assert_almost_equal, assert_equal

import sys
sys.path.append("../sim_utils/")
from dgp import linear_model


def _compute_all_residuals(X, alphas, gram, max_iter=5000, tol=1e-3,
                           method='lasso', n_jobs=1, verbose=0):
    """Nodewise Lasso. Compute all the residuals: regressing each column of the
    design matrix against the other columns"""

    n_samples, n_features = X.shape

    results = \
        Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_compute_residuals)
                (X=X,
                 column_index=i,
                 alpha=alphas[i],
                 gram=gram,
                 max_iter=max_iter,
                 tol=tol,
                 method=method)
            for i in range(n_features))

    results = np.asarray(results)
    Z = np.stack(results[:, 0], axis=1)
    omega_diag = np.stack(results[:, 1])

    return Z, omega_diag


def _compute_residuals(X, column_index, alpha, gram, max_iter=5000,
                       tol=1e-3, method='lasso'):
    """Compute the residuals of the regression of a given column of the
    design matrix against the other columns"""

    n_samples, n_features = X.shape
    i = column_index

    X_new = np.delete(X, i, axis=1)
    y = np.copy(X[:, i])

    if method == 'lasso':

        gram_ = np.delete(np.delete(gram, i, axis=0), i, axis=1)
        clf = Lasso(alpha=alpha, precompute=gram_, max_iter=max_iter, tol=tol)

    else:

        ValueError("The only regression method available is 'lasso'")

    clf.fit(X_new, y)
    z = y - clf.predict(X_new)

    omega_diag_i = n_samples * np.sum(z ** 2) / np.dot(y, z) ** 2

    return z, omega_diag_i


def desparsified_lasso(X, y, dof_ajdustement=False,
                       confidence=0.95, max_iter=5000, tol=1e-3,
                       residual_method='lasso', alpha_max_fraction=0.01,
                       n_jobs=1, memory=None, verbose=0):

    """Desparsified Lasso with confidence intervals

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    dof_ajdustement : bool, optional (default=False)
        If True, makes the degrees of freedom adjustement (cf. [4]_ and [5]_).
        Otherwise, the original Desparsified Lasso estimator is computed
        (cf. [1]_ and [2]_ and [3]_).

    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].

    max_iter : int, optional (default=5000)
        The maximum number of iterations when regressing, by Lasso,
        each column of the design matrix against the others.

    tol : float, optional (default=1e-3)
        The tolerance for the optimization of the Lasso problems: if the
        updates are smaller than `tol`, the optimization code checks the
        dual gap for optimality and continues until it is smaller than `tol`.

    residual_method : str, optional (default='lasso')
        Method used for computing the residuals of the Nodewise Lasso.
        Currently the only method available is 'lasso'.

    alpha_max_fraction : float, optional (default=0.01)
        Only used if method='lasso'.
        Then alpha = alpha_max_fraction * alpha_max.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the Nodewise Lasso.

    memory : str or joblib.Memory object, optional (default=None)
        Used to cache the output of the computation of the Nodewise Lasso.
        By default, no caching is done. If a string is given, it is the path
        to the caching directory.

    verbose: int, optional (default=1)
        The verbosity level: if non zero, progress messages are printed
        when computing the Nodewise Lasso in parralel.
        The frequency of the messages increases with the verbosity level.

    Returns
    -------
    beta_hat : array, shape (n_features,)
        Estimated parameter vector.

    cb_min : array, shape (n_features)
        Lower bound of the confidence intervals on the parameter vector.

    cb_max : array, shape (n_features)
        Upper bound of the confidence intervals on the parameter vector.

    References
    ----------
    .. [1] Zhang, C. H., & Zhang, S. S. (2014). Confidence intervals for
           low dimensional parameters in high dimensional linear models.
           Journal of the Royal Statistical Society: Series B: Statistical
           Methodology, 217-242.

    .. [2] Van de Geer, S., Bühlmann, P., Ritov, Y. A., & Dezeure, R. (2014).
           On asymptotically optimal confidence regions and tests for
           high-dimensional models. Annals of Statistics, 42(3), 1166-1202.

    .. [3] Javanmard, A., & Montanari, A. (2014). Confidence intervals and
           hypothesis testing for high-dimensional regression. The Journal
           of Machine Learning Research, 15(1), 2869-2909.

    .. [4] Bellec, P. C., & Zhang, C. H. (2019). De-biasing the lasso with
           degrees-of-freedom adjustment. arXiv preprint arXiv:1902.08885.

    .. [5] Celentano, M., Montanari, A., & Wei, Y. (2020). The Lasso with
           general Gaussian designs with applications to hypothesis testing.
           arXiv preprint arXiv:2007.13716.
    """

    X = np.asarray(X)

    n_samples, n_features = X.shape

    memory = check_memory(memory)

    y = y - np.mean(y)
    X = X - np.mean(X, axis=0)
    gram = np.dot(X.T, X)
    gram_nodiag = gram - np.diag(np.diag(gram))

    list_alpha_max = np.max(np.abs(gram_nodiag), axis=0) / n_samples
    alphas = alpha_max_fraction * list_alpha_max

    # Calculating precision matrix (Nodewise Lasso)
    Z, omega_diag = memory.cache(_compute_all_residuals, ignore=['n_jobs'])(
        X, alphas, gram, max_iter=max_iter, tol=tol,
        method=residual_method, n_jobs=n_jobs, verbose=verbose)

    # Lasso regression
    sigma_hat, beta_lasso = reid(X, y, n_jobs=n_jobs)

    # Computing the degrees of freedom adjustement
    if dof_ajdustement:
        coef_max = np.max(np.abs(beta_lasso))
        support = np.sum(np.abs(beta_lasso) > 0.01 * coef_max)
        support = min(support, n_samples - 1)
        dof_factor = n_samples / (n_samples - support)
    else:
        dof_factor = 1

    # Computing Desparsified Lasso estimator and confidence intervals
    beta_bias = dof_factor * np.dot(y.T, Z) / np.sum(X * Z, axis=0)

    P = ((Z.T.dot(X)).T / np.sum(X * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))
    Id = np.identity(n_features)
    P_nodiag = dof_factor * P_nodiag + (dof_factor - 1) * Id

    beta_hat = beta_bias - P_nodiag.dot(beta_lasso)

    omega_diag = omega_diag * dof_factor ** 2
    omega_invsqrt_diag = omega_diag ** (-0.5)

    quantile = stats.norm.ppf(1 - (1 - confidence) / 2)

    confint_radius = np.abs(quantile * sigma_hat /
                            (np.sqrt(n_samples) * omega_invsqrt_diag))
    cb_max = beta_hat + confint_radius
    cb_min = beta_hat - confint_radius

    return beta_hat, cb_min, cb_max




if __name__ == '__main__':
    n_samples, n_features = 100, 10
    support_size = 1
    sigma = 0.01
    rho = 0.0

    X = np.random.normal(size=(n_samples,n_features))
    y,support,beta_s = linear_model(X=X,sigma=sigma,s=support_size,beta=1.0,return_support=True)
    beta_hat, cb_min, cb_max = desparsified_lasso(X, y)
    beta = np.zeros(n_features)
    for j in range(support_size):
        beta[j] = beta_s[j]
    print("beta_hat")
    print(beta_hat)
    print("lower confidence interval")
    print(cb_min)
    print("upper confidence interval")
    print(cb_max)

    assert_almost_equal(beta_hat, beta, decimal=1)
    assert_equal(cb_min < beta, True)
    assert_equal(cb_max > beta, True)
