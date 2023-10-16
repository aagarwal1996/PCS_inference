import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score

models = {"OLS": LinearRegression, "RidgeCV": RidgeCV}  # "LassoCV": LassoCV}


def predictability_screening(X_train, y_train, X_val, y_val):
    prediction_performance = {}
    for name, m in models.items():
        cls = m()
        cls.fit(X_train, y_train)
        prediction_performance[name] = r2_score(y_val, cls.predict(X_val))
    return prediction_performance


def _fit_model_perturbed_datasets(model_name, X_train_perturbed_list, y_train_perturbed_list):
    model_coefficients = np.zeros((len(X_train_perturbed_list), X_train_perturbed_list[0].shape[1]))
    for i in range(len(X_train_perturbed_list)):
        X_perturbed, y_perturbed = X_train_perturbed_list[i], y_train_perturbed_list[i]
        perturbed_model = models[model_name]().fit(X_perturbed, y_perturbed)
        model_coefficients[i, :] = perturbed_model.coef_
    return model_coefficients


def fit_all_model_perturbed_datasets(p_screened_model_names, X_train_perturbed_list, y_train_perturbed_list):
    p_screened_coefficients = {}
    for name in p_screened_model_names:
        p_screened_coefficients[name] = _fit_model_perturbed_datasets(name, X_train_perturbed_list,
                                                                      y_train_perturbed_list)
    return p_screened_coefficients


def compute_confidence_intervals(X_train, p_screened_coefficients, strategy='LCB', conf_level=0.05):
    confidence_intervals = np.zeros((X_train.shape[1], 2))
    num_features = X_train.shape[1]
    for k in range(num_features):
        all_model_coef = []
        for name, coef in p_screened_coefficients.items():
            all_model_coef.append(p_screened_coefficients[name][:, k])
        all_model_coef = np.concatenate(all_model_coef)
        k_confidence_interval_lower = np.quantile(all_model_coef, conf_level / 2)
        k_confidence_interval_upper = np.quantile(all_model_coef, 1 - conf_level / 2)
        confidence_intervals[k, 0] = k_confidence_interval_lower
        confidence_intervals[k, 1] = k_confidence_interval_upper
    return confidence_intervals

# def fit_models(X_train_perturbed_list,y_train_perturbed_list, models):
#    for i in range(len(X_train_perturbed_list)):
#        X_perturbed,y_perturbed  = X_train_perturbed_list[i], y_train_perturbed_list[i]
#        for m in models:
#            cls = m()
#            cls.fit(X_perturbed, y_perturbed)
