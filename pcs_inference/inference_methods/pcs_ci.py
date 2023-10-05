import logging

import numpy as np

from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
class PCS_Infernece(object):
    def __init__(self, methods, p_screen_threshold, metric):
        self.methods = methods
        self.p_screen_threshold = p_screen_threshold
        self.metric = metric

    def _eval_method(self, method, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # fit model
        method.fit(X_train, y_train)
        # evaluate model
        yhat = method.predict(X_test)
        # evaluate predictions
        score = self.metric(y_test, yhat)
        return score

    def p_screen(self, X, y):
        # split data into training and test sets
        methods_performance = [self._eval_method(m(), X, y) for m in self.methods]
        LOGGER.info(f"methods performance: {methods_performance}")
        methods_filtered = [m for i, m in enumerate(self.methods) if methods_performance[i] > self.p_screen_threshold]
        logging.info(f"methods passed filters: {[m.__name__ for m in methods_filtered]}")
        return methods_filtered

    def generate_data(self, methods, X, residuals, n_bootstrap=100):
        # take a bootstrap sample from X and predict y using each method
        Xs, ys = [], []
        size = n_bootstrap // len(methods)
        sd = np.std(residuals) * 5
        for method in methods:
            X_bs = X[np.random.choice(X.shape[0], size=size, replace=True), :]
            y_bs = method.predict(X_bs) + np.random.normal(loc=0, scale=sd, size=size)
            Xs.append(X_bs)
            ys.append(y_bs)
        # concatenate Xs and ys
        Xs = np.concatenate(Xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        return Xs, ys

    def calc_ci(self, X, y, n_reps=10, n_bootstrap=100, alpha=0.05):
        methods_filtered = self.p_screen(X, y)
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        methods_fitted = [m().fit(X_train, y_train) for m in methods_filtered]
        test_set_residuals = [y_test - m.predict(X_test) for m in methods_fitted]
        # make it into one vector
        residuals = np.concatenate(test_set_residuals, axis=0)
        coefficients = []
        for i in range(n_reps):
            X, y = self.generate_data(methods_fitted, X, residuals, n_bootstrap)
            coefficients += [m().fit(X, y).coef_ for m in methods_filtered]
        # take an interval for each coefficient index
        coef_mat = np.array(coefficients)
        p = 100 * alpha / 2
        lower = np.percentile(coef_mat, p, axis=0)
        upper = np.percentile(coef_mat, 100 - p, axis=0)
        return lower, upper
