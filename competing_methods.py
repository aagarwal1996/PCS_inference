import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.utils import resample
import scipy.stats as st
from sklearn.linear_model import LinearRegression



def classic_confidence_interval(X,y,conf_level = 0.05):
    X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()
    return res.conf_int(conf_level)

def bootstrap_confidence_interval(X,y,conf_level = 0.05,num_iterations = 50):
    model_coefs = np.zeros((num_iterations,X.shape[1]))
    for i in range(num_iterations):
        X_bootstrap, y_bootstrap = resample(X,y)
        m = LinearRegression().fit(X_bootstrap,y_bootstrap)
        model_coefs[i,:] = m.coef_
    confidence_intervals = []
    for k in range(X.shape[1]):
        confidence_intervals.append(st.norm.interval(alpha= 1 - conf_level, loc = np.mean(model_coefs[:,k]), scale = st.tstd(model_coefs[:,k])))
    return confidence_intervals
        

    