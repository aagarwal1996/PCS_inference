import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.utils import resample
import scipy.stats as st
from sklearn.linear_model import LinearRegression

import sys
sys.path.append("../sim_utils/")
from dgp import linear_model
from evaluation_metrics import *



def classic_confidence_interval(X,y,conf_level = 0.05):
    X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()
    cbs = res.conf_int(conf_level)
    return cbs[1:,0],cbs[1:,1]



if __name__ == '__main__':
    n_samples, n_features = 100, 10
    support_size = 2
    sigma = 0.01
    rho = 0.0

    X = np.random.normal(size=(n_samples,n_features))
    y,support,beta_s = linear_model(X=X,sigma=sigma,s=support_size,beta=1.0,return_support=True)
    cb_min,cb_max = classic_confidence_interval(X,y)
    print(cb_min,cb_max)