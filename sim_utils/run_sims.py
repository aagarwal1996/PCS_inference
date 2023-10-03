import numpy as np
from numpy.linalg import multi_dot
from scipy import stats
from scipy.linalg import inv
from joblib import Parallel, delayed
from sklearn.utils.validation import check_memory
from sklearn.linear_model import LassoCV,LinearRegression
import statsmodels.api as sm
from numpy.testing import assert_almost_equal, assert_equal
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from evaluation_metrics import *
from dgp import *


import sys
sys.path.append("../inference_methods/")
from classic_confidence_interval import *
from desparsified_lasso_inference import *
from empirical_bootstrap import *
from residual_bootstrap import *
from lasso_mls import *

inference_methods = ["Classic_CI","Empirical_Bootstrap", "Residual_Bootstrap","Desparsified_Lasso","lasso_mls"]

def run_sim(X,sigma,s,beta,n_repeats):
    """
    Given X,sigma,s,beta
        1. generates synthetic data 
        2. runs various inference methods
        3. computes average coverage and average length of intervals for each method over n_repeats
    """
    coverage = {} #initialize 
    lengths = {}
    for method in inference_methods:
        coverage[method] = np.zeros((n_repeats,X.shape[1]))
        lengths[method] = np.zeros((n_repeats,X.shape[1]))
        
    for i in range(n_repeats):
        y,support,beta_s = linear_model(X=X,sigma=sigma,s=support_size,beta = beta,return_support=True)
        for method in inference_methods:
            if method == "Classic_CI":
                cb_min,cb_max = classic_confidence_interval(X,y)
            elif method == "Empirical_Bootstrap":
                cb_min,cb_max = empirical_bootstrap(X,y)
            elif method == "Residual_Bootstrap":
                cb_min,cb_max = residual_bootstrap(X,y)
            elif method == "Desparsified_Lasso":
                beta_hat,cb_min,cb_max = desparsified_lasso(X,y)
            elif method == "lasso_mls":
                cb_min,cb_max = residual_mls(X,y)
            lengths[method][i,:] = np.array(get_length(cb_min,cb_max))
            coverage[method][i,:] = np.array(check_if_covers(beta_s,cb_min,cb_max))
    for key in coverage:
        coverage[key] = np.mean(coverage[key],axis = 0)
        lengths[key] = np.mean(lengths[key],axis = 0)

    return coverage,lengths


if __name__ == '__main__':
    n_samples, n_features = 100, 3
    support_size = 2
    sigma = 0.01
    rho = 0.0

    X = np.random.normal(size=(n_samples,n_features))
    coverage,lengths = run_sim(X,sigma,support_size,1.0,3)
    print(coverage["Classic_CI"],lengths["Classic_CI"])

            
                

 