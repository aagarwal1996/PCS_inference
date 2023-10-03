import numpy as np
from numpy.linalg import multi_dot
from scipy import stats
from scipy.linalg import inv
from joblib import Parallel, delayed
from sklearn.utils.validation import check_memory
from sklearn.linear_model import LassoCV,LinearRegression
import statsmodels.api as sm
from noise_estimation import reid
from numpy.testing import assert_almost_equal, assert_equal
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import sys
sys.path.append("../sim_utils/")
from dgp import linear_model
from evaluation_metrics import *



def residual_bootstrap(X,y,model = 'linear', B = 50):
    """
    Compute confidence intervals via residual bootstrap.
    
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
    cb_min = [0]*X.shape[1]
    cb_max = [0]*X.shape[1]
    if model == 'linear':
        reg = LinearRegression()
    elif model == 'lasso':
        reg = LassoCV()
    else: 
        raise ValueError('The only regression method available is lasso and OLS')

    reg.fit(X,y) #fit model 
    preds = reg.predict(X)
    residuals = y - preds #compute residuals
    residuals = residuals - np.mean(residuals) #center residuals
    model_coefs = np.zeros((B,X.shape[1])) #store coefficients across bootstrap samples
    for i in range(B): #compute bootstrap residuals and refit model.
        bootstrapped_indices = resample([*range(n)])
        X_resampled = X[bootstrapped_indices,:]
        y_resampled = preds[bootstrapped_indices] + residuals[bootstrapped_indices]
        if model == 'linear':
            bootstrapped_model = LinearRegression()
        if model == 'lasso':
             bootstrapped_model = LassoCV()
        bootstrapped_model.fit(X_resampled,y_resampled)
        model_coefs[i,:] = bootstrapped_model.coef_
    stds = np.std(model_coefs,axis = 0)
    cb_min =  reg.coef_ - 1.96*stds
    cb_max = reg.coef_ + 1.96*stds
    return cb_min,cb_max
        
    
        


if __name__ == '__main__':
    n_samples, n_features = 100, 10
    support_size = 2
    sigma = 0.01
    rho = 0.0

    X = np.random.normal(size=(n_samples,n_features))
    y,support,beta_s = linear_model(X=X,sigma=sigma,s=support_size,beta=1.0,return_support=True)
    cb_min,cb_max = residual_bootstrap(X,y)
    print(cb_min,cb_max)
    print(get_length(cb_min,cb_max))
    print(check_if_covers(beta_s,cb_min,cb_max))
    