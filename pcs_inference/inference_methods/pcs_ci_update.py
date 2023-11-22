import logging

import numpy as np
from sklearn.utils import resample
import pprint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

import sys




def generate_bootstrap_oob_indices(n):
    '''
    Generates bootstrap samples 
    '''
    # Generate bootstrap indices
    bootstrap_indices = np.random.choice(n, size=n, replace=True)

    # Find out-of-bag indices
    oob_indices = np.setdiff1d(np.arange(n), bootstrap_indices)

    return bootstrap_indices, oob_indices

def get_coverage(test_coef_feature,ci):
    '''
    get coverage of confidence interval for a given feature
    '''
    coverage = 0
    for bootstrapped_coef in test_coef_feature:
        if ci[0] <= bootstrapped_coef <= ci[1]:
            coverage += 1
    return coverage/len(test_coef_feature)
    
    
    

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PCSInference(object):
    def __init__(self, methods, p_screen_threshold, metric):
        self.methods = methods
        self.p_screen_threshold = p_screen_threshold
        self.metric = metric
        self.filtered_methods = []

    def p_screen(self,X,y,n_bootstrap = 100):
        '''
        Does p-screening on out of bag samples. 
        '''
        num_methods = len(self.methods)
        
        errors = np.zeros((num_methods, n_bootstrap))
        coef_ = np.zeros((n_bootstrap,X.shape[1],num_methods))
        filtered_methods = []
        for b in range(n_bootstrap):
            bootstrap_indices, oob_indices = generate_bootstrap_oob_indices(len(X))
            X_in_bag,y_in_bag = X[bootstrap_indices,:],y[bootstrap_indices]
            X_oob,y_oob = X[oob_indices,:],y[oob_indices]
            for i,method in enumerate(self.methods):
                m = method()
                m.fit(X_in_bag,y_in_bag)
                methods_preds = m.predict(X_oob)
                method_error = self.metric(methods_preds,y_oob)
                errors[i,b] = method_error
                coef_[b,:,i] = m.coef_
        average_error = np.mean(errors,axis = 1)
        filtered_methods = []
        filtered_method_indices = []
        for i,err in enumerate(average_error): 
            if err < self.p_screen_threshold:
                filtered_methods.append(self.methods[i])
                filtered_method_indices.append(i)
        self.filtered_methods = filtered_methods 
        filtered_coefs = coef_[:,:,filtered_method_indices]
        return filtered_methods,filtered_coefs

    def get_test_coefs(self,X_test,n_bootstrap,coefs,residuals):
        '''
        Get test coefficients to assess coverage
        '''
        test_coefs = np.zeros((n_bootstrap,X_test.shape[1],len(self.filtered_methods)))
        for b in range(n_bootstrap):
            X_synthetic, y_synthetic = self.generate_data(X_test,coefs,residuals)
            for i,method in enumerate(self.filtered_methods):
                m =  method()
                m.fit(X_synthetic, y_synthetic)
                test_coefs[b,:,i] = m.coef_
        return test_coefs

    def get_adjusted_interval(self,test_coefs,lower,upper,alpha,expansion_percentage,linear_search_steps = 2000):
        '''
        adjust interval according to empirical coverage
        '''
        concatenated_bounds = [[lower[i],upper[i]] for i in range(len(upper))]
        adjusted_bounds = []
        for i in range(len(concatenated_bounds)):
            confidence_intervals_i = concatenated_bounds[i]
            test_coefs_feature_i = test_coefs[:,i,:].flatten() # is of length num_filtered_methods * num_bootstraps
            if get_coverage(test_coefs_feature_i,confidence_intervals_i) >= 1.0 - alpha:
                adjusted_bounds.append(confidence_intervals_i)
            else:        #adjust confidence interval via binary search on length
                orig_lower_i,orig_upper_i= confidence_intervals_i[0],confidence_intervals_i[1]
                orig_confidence_interval_i = [orig_lower_i,orig_upper_i]
                orig_length_i = confidence_intervals_i[1] - confidence_intervals_i[0]
                while get_coverage(test_coefs_feature_i,confidence_intervals_i) < 1.0 - alpha: #get upper bound on length 
                    cur_length = confidence_intervals_i[1] - confidence_intervals_i[0]
                    expansion_amt = cur_length * expansion_percentage
                    confidence_intervals_i[0] = confidence_intervals_i[0] - expansion_amt/2
                    confidence_intervals_i[1] = confidence_intervals_i[1] + expansion_amt/2
                upper_bound_length_i = confidence_intervals_i[1] - confidence_intervals_i[0]
                step_size = (upper_bound_length_i - orig_length_i)/linear_search_steps
                while get_coverage(test_coefs_feature_i,orig_confidence_interval_i) < 1.0 - alpha:
                    orig_confidence_interval_i[0] -= step_size
                    orig_confidence_interval_i[1] += step_size
            
                adjusted_bounds.append(orig_confidence_interval_i)
        return adjusted_bounds

    def generate_data(self, X_test, coefs,residuals):
        X_synthetic = []
        y_synthetic = []
        size = len(X_test) // len(coefs)
        for i,method in enumerate(self.filtered_methods):
            bootstrap_indices = np.random.choice(X_test.shape[0], size=size, replace=True)
            X_bs = X_test[bootstrap_indices, :]
            residual_bs = residuals[i,bootstrap_indices]
            method_coef = coefs[i,:]
            y_bs =  np.matmul(X_bs,coef) + residual_bs
            X_synthetic.append(X_bs)
            y_synthetic.append(y_bs)
        # concatenate Xs and ys
        X_synthetic = np.concatenate(X_synthetic, axis=0)
        y_synthetic = np.concatenate(y_synthetic, axis=0)
        return X_synthetic, y_synthetic

    def calc_ci(self, X, y, n_bootstrap=100, alpha=0.05,test_size = 0.5,expansion_percentage = 0.001):
        '''
        Does p-screening and computes confidence intervals
        '''
        
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=42)
        
        #return methods that pass p-check and filtered coefficients
        filtered_methods,filtered_coefs = self.p_screen(X_train, y_train,n_bootstrap = n_bootstrap) #filtered coefs has shape num_bootstrap,X.shape[1],num_filtered_methods
       

        num_filtered_methods = filtered_coefs.shape[2]
        if num_filtered_methods == 0: #
            raise ValueError("No methods passed p-check")

        filtered_coefs_flattened = filtered_coefs.transpose(0, 2, 1).reshape(n_bootstrap*num_filtered_methods, X.shape[1]) #has shape num_bootstrap*num_filtered_methods,X.shape[1]

        p = 100 * alpha / 2
        lower = np.percentile(filtered_coefs_flattened, p, axis=0)
        upper = np.percentile(filtered_coefs_flattened, 100 - p, axis=0)

        coefs = np.zeros((num_filtered_methods,X.shape[1]))
        residuals = np.zeros((num_filtered_methods,X_train.shape[0]))
        for i,method in enumerate(self.filtered_methods):
            m = method()
            m.fit(X_train,y_train)
            y_train_pred = m.predict(X_train)
            coefs[i][:] = m.coef_
            residuals[i][:] = y_train - y_train_pred
        
        test_coefs =  self.get_test_coefs(X_test,n_bootstrap,coefs,residuals)
        adjusted_bounds = self.get_adjusted_interval(test_coefs,lower,upper,alpha,expansion_percentage)
        cb_min = [adjusted_bounds[i][0] for i in range(len(adjusted_bounds))]
        cb_max = [adjusted_bounds[i][1] for i in range(len(adjusted_bounds))]
        return cb_min,cb_max
    
    

    


if __name__ == "__main__":
    X, y,coef = make_regression(n_samples=100, n_features=20, noise=1.0, tail_strength = 1.0,random_state=42,coef = True )
    pcs_i = PCSInference(methods = [LinearRegression,RidgeCV,LassoCV],p_screen_threshold = 10.0,metric = mean_squared_error)
    cb_min,cb_max = pcs_i.calc_ci(X,y)
    pcs_bounds = [(cb_min[i],cb_max[i]) for i in range(len(cb_max))]
    print(f"confidence bounds:  {pcs_bounds}")

    print("")
    print("")
   
    print(f"True coefficients: {coef} ")

                         