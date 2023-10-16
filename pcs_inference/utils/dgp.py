import numpy as np
import random



def generate_coef(beta, s):
    if isinstance(beta, int) or isinstance(beta, float):
        beta = np.repeat(beta, repeats=s)
    return beta


def linear_model(X, sigma, s, beta, heritability=None, snr=None, error_fun=None,
                 frac_corrupt=None, corrupt_how='permute', corrupt_size=None, 
                 corrupt_mean=None, return_support=False):
    """
    This method is used to crete responses from a linear model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity
    beta: coefficient vector. If beta not a vector, then assumed a constant
    sigma: s.d. of added noise
    Returns:
    numpy array of shape (n)
    """
    n, p = X.shape
    def create_y(x, s, beta):
        linear_term = 0
        for j in range(s):
            linear_term += x[j] * beta[j]
        return linear_term

    beta = generate_coef(beta, s)
    y_train = np.array([create_y(X[i, :], s, beta) for i in range(len(X))])
    return y_train, beta
    if heritability is not None:
        sigma = (np.var(y_train) * ((1.0 - heritability) / heritability)) ** 0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr) ** 0.5
    if error_fun is None:
        error_fun = np.random.randn
    if frac_corrupt is None and corrupt_size is None:
        y_train = y_train + sigma * error_fun(n) #size =
    else:
        if frac_corrupt is None:
            frac_corrupt = 0
        num_corrupt = int(np.floor(frac_corrupt*len(y_train)))
        corrupt_indices = random.sample([*range(len(y_train))], k=num_corrupt)
        if corrupt_how == 'permute':
            corrupt_array = y_train[corrupt_indices]
            corrupt_array = random.sample(list(corrupt_array), len(corrupt_array))
            for i,index in enumerate(corrupt_indices):
                y_train[index] = corrupt_array[i]
            y_train = y_train + sigma * error_fun(n)           
        elif corrupt_how == 'cauchy':
            for i in range(len(y_train)):
                if i in corrupt_indices:
                    y_train[i] = y_train[i] + sigma*np.random.standard_cauchy()
                else:
                     y_train[i] = y_train[i] + sigma*error_fun()
        elif corrupt_how == "leverage_constant":
            if isinstance(corrupt_size, int):
                corrupt_quantile = corrupt_size / n
            else:
                corrupt_quantile = corrupt_size
            y_train = y_train + sigma * error_fun(n)
            corrupt_idx = np.random.choice(range(s, p), size=1)
            #y_train = corrupt_leverage(X[:, corrupt_idx], y_train, mean_shift=corrupt_mean, corrupt_quantile=corrupt_quantile, mode="constant")
        elif corrupt_how == "leverage_normal":
            if isinstance(corrupt_size, int):
                corrupt_quantile = corrupt_size / n
            else:
                corrupt_quantile = corrupt_size
            y_train = y_train + sigma * error_fun(n)
            corrupt_idx = np.random.choice(range(s, p), size=1)
            #y_train = corrupt_leverage(X[:, corrupt_idx], y_train, mean_shift=corrupt_mean, corrupt_quantile=corrupt_quantile, mode="normal")

    if return_support:
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        beta_full = np.zeros(X.shape[1])
        for i in range(s):
            beta_full[i] = beta[i]
        return y_train, support, beta_full
    else:
        return y_train
