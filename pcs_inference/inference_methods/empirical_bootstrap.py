import numpy as np
import scipy.stats as st

from sklearn.linear_model import LassoCV,LinearRegression

from sklearn.utils import resample

from pcs_inference.utils.evaluation_metrics import check_if_covers, get_length
from ..utils.dgp import linear_model

def empirical_bootstrap(X,y,conf_level, model = 'linear', B = 50):
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

    model_coefs = np.zeros((B,X.shape[1])) #compute bootstrap residuals and refit model.
    reg.fit(X,y) #fit model 
    for i in range(B):
        bootstrapped_indices = resample([*range(n)])
        X_resampled = X[bootstrapped_indices,:]
        y_resampled = y[bootstrapped_indices]
        if model == 'linear':
            bootstrapped_model = LinearRegression()
        if model == 'lasso':
             bootstrapped_model = LassoCV()
        bootstrapped_model.fit(X_resampled,y_resampled)
        model_coefs[i,:] = bootstrapped_model.coef_
    stds = np.std(model_coefs,axis = 0)
    gaussian_quantile = st.norm.ppf(1 - conf_level / 2)
    cb_min =  reg.coef_ - gaussian_quantile*stds
    cb_max = reg.coef_ + gaussian_quantile*stds
    return cb_min,cb_max
        
    
        


if __name__ == '__main__':
    n_samples, n_features = 100, 10
    support_size = 2
    sigma = 0.01
    rho = 0.0

    X = np.random.normal(size=(n_samples,n_features))
    y,support,beta_s = linear_model(X=X,sigma=sigma,s=support_size,beta=1.0,return_support=True)
    cb_min,cb_max = empirical_bootstrap(X,y)
    print(cb_min,cb_max)
    print(get_length(cb_min,cb_max))
    print(check_if_covers(beta_s,cb_min,cb_max))