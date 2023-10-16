import os

import numpy as np

from collections import namedtuple

import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, LassoLars, ElasticNetCV, LassoLarsCV
from sklearn.metrics import r2_score

from pcs_inference.inference_methods.ci import confidence_interval
from pcs_inference.inference_methods.pcs_ci import PCS_Infernece
from pcs_inference.utils.dgp import linear_model
from pcs_inference.utils.evaluation_metrics import get_length, check_if_covers

# inference_methods = ["Classic_CI", "Empirical_Bootstrap", "Residual_Bootstrap", "Desparsified_Lasso", "lasso_mls"]
inference_methods = ["bootstrap", "empirical_bootstrap", "mls"]


# create a namedtuple for inference methods
# inference_methods = namedtuple('inference_methods', ['Classic_CI', 'Empirical_Bootstrap', 'Residual_Bootstrap', 'Desparsified_Lasso', 'lasso_mls'])


def run_sim(X, y, beta, n_repeats, alpha):
    """
    Given X,sigma,s,beta
        1. generates synthetic data 
        2. runs various inference methods
        3. computes average coverage and average length of intervals for each method over n_repeats
    """

    pcs = PCS_Infernece(methods=[LassoCV, RidgeCV, LassoLarsCV, ElasticNetCV],
                        p_screen_threshold=0.1, metric=r2_score)
    results = {method: {"coverage": [], "length": []} for method in inference_methods}
    results["PCS"] = {"coverage": [], "length": []}
    non_zero_idx = beta != 0
    beta_non_zero = beta[non_zero_idx]
    for i in range(n_repeats):
        for method in inference_methods:

            cb_min, cb_max = confidence_interval(X, y, method=method, alpha=alpha)
            cb_min = np.array(cb_min)[non_zero_idx]
            cb_max = np.array(cb_max)[non_zero_idx]
            results[method]["coverage"].append(np.mean(check_if_covers(beta_non_zero, cb_min, cb_max)))
            results[method]["length"].append(np.mean(get_length(cb_min, cb_max)))
        cb_min, cb_max = pcs.calc_ci(X, y, alpha=alpha)
        cb_min = np.array(cb_min)[non_zero_idx]
        cb_max = np.array(cb_max)[non_zero_idx]

        results["PCS"]["coverage"].append(np.mean(check_if_covers(beta_non_zero, cb_min, cb_max)))
        # print(np.mean(get_length(cb_min, cb_max)))
        results["PCS"]["length"].append(np.mean(get_length(cb_min, cb_max)))

    # save this dataframe as figure

    return results


if __name__ == '__main__':
    n_samples, n_features = 100, 20
    support_size = 3
    snr = 10
    results_dir = "results"
    alpha = 0.05
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for noise in ["gaussian", "laplace", "cauchy"]:
        for design in ["gaussian", "enhancer"]:

            if design == "gaussian":
                X = np.random.normal(size=(n_samples, n_features))
                beta = np.zeros(n_features)

            elif design == "enhancer":
                X = pd.read_csv("data/X_uncorrelated_enhancer.csv").values
                # remove rows with nan
                X = X[~np.isnan(X).any(axis=1)]
                n_samples = X.shape[0]
                n_features = X.shape[1]
                beta = np.zeros(n_features)
            for j in np.random.choice(n_features, support_size):
                beta[j] = np.random.normal()
            sim_name = f"{noise}_noise_n{n_samples}_p{n_features}_s{support_size}_snr{snr}_alpha{alpha}_design{design}"

            # generate a random coefficient vector

            # normalize beta
            beta = beta / np.linalg.norm(beta)
            y = X @ beta


            if noise== "laplace":
                y+= (1 / snr) * np.random.standard_cauchy(size=n_samples)
            elif noise == "gaussian":
                y += (1 / snr) * np.random.normal(size=n_samples)
            elif noise == "cauchy":
                y+= (1 / snr) * np.random.standard_cauchy(size=n_samples)

            beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
            try:

                results = run_sim(X, y, beta, n_repeats=10, alpha=alpha)
                results_summary = pd.DataFrame({method: {"coverage (mean)": np.mean(results[method]["coverage"]),
                                                         "coverage (std)": np.std(results[method]["coverage"]),
                                                         "length (mean)": np.mean(results[method]["length"]),
                                                         "length (std)": np.std(results[method]["length"])
                                                         } for method in inference_methods + ["PCS"]})

                results_summary.round(3).to_csv(os.path.join(results_dir, f"{sim_name}.csv"))
            except Exception:
                print(f"failed for {sim_name}")
                continue
