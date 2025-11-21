import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import KFold
import pymc as pm

class BinaryClassifier:
    def __init__(self,args, X, y, n_splits=10, random_state=42):
        self.args = args
        self.X = X
        self.y_original = y
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.y_binary, self.weights = self._prepare_target()

    def _prepare_target(self):
        y_str = self.y_original.apply(lambda val: 'Normal' if str(val).startswith('C') else ('Alzheimer' if str(val).startswith('AD') else ('MCI')))
        y_binary = y_str.apply(lambda val: 1 if val == 'Alzheimer' else 0)
        weights = np.ones(len(y_str))
        mci_weight = 0.8
        weights[y_str[y_str == 'MCI'].index] = mci_weight
        return y_binary, weights

    def cv_logistic_regression(self):
        acc_scores, recall_scores, f1_scores = [], [], []

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y_binary.iloc[train_index], self.y_binary.iloc[val_index]
            weights_train, weights_val = self.weights[train_index], self.weights[val_index]

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = model.predict(X_val)

            acc_scores.append(accuracy_score(y_val, y_pred, sample_weight=weights_val))
            recall_scores.append(recall_score(y_val, y_pred, sample_weight=weights_val, zero_division=0))
            f1_scores.append(f1_score(y_val, y_pred, sample_weight=weights_val, zero_division=0))

        return {'accuracy': np.mean(acc_scores), 'recall': np.mean(recall_scores), 'f1_score': np.mean(f1_scores)}

    def cv_bayesian_logistic_regression(self):
        acc_scores, recall_scores, f1_scores = [], [], []
        n_features = self.X.shape[1]

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y_binary.iloc[train_index], self.y_binary.iloc[val_index]
            weights_train, weights_val = self.weights[train_index], self.weights[val_index]

            with pm.Model() as weighted_logistic_model:
                alpha = pm.Normal("alpha", mu=0, sigma=1)
                beta = pm.Laplace("beta", mu=0, b=self.args.binary_b, shape=n_features)
                logit_p = alpha + pm.math.dot(X_train, beta)
                log_likelihood = pm.logp(pm.Bernoulli.dist(logit_p=logit_p), y_train)
                weighted_log_likelihood = log_likelihood * weights_train
                pm.Potential('weighted_likelihood', weighted_log_likelihood.sum())
                trace = pm.sample(500, tune=500, cores=1, progressbar=False, return_inferencedata=False)

            alpha_samples = trace['alpha']
            beta_samples = trace['beta']
            all_probs = [1 / (1 + np.exp(-(alpha_s + np.dot(X_val, beta_s)))) for alpha_s, beta_s in zip(alpha_samples, beta_samples)]
            y_prob_bayesian = np.mean(all_probs, axis=0)
            y_pred_bayesian = (y_prob_bayesian > 0.5).astype(int)

            acc_scores.append(accuracy_score(y_val, y_pred_bayesian, sample_weight=weights_val))
            recall_scores.append(recall_score(y_val, y_pred_bayesian, sample_weight=weights_val, zero_division=0))
            f1_scores.append(f1_score(y_val, y_pred_bayesian, sample_weight=weights_val, zero_division=0))

        return {'accuracy': np.mean(acc_scores), 'recall': np.mean(recall_scores), 'f1_score': np.mean(f1_scores)}

    def fit_final_models(self, draws=500, tune=500):
        """
        Fits both standard and Bayesian logistic regression models on the entire dataset.
        """
        # Standard Logistic Regression
        self.final_logistic_model = LogisticRegression(max_iter=1000)
        self.final_logistic_model.fit(self.X, self.y_binary, sample_weight=self.weights)

        # Bayesian Logistic Regression
        n_features = self.X.shape[1]
        
        with pm.Model() as self.final_bayesian_model:
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            beta = pm.Laplace("beta", mu=0, b=self.args.binary_b, shape=n_features)
            logit_p = alpha + pm.math.dot(self.X, beta)
            log_likelihood = pm.logp(pm.Bernoulli.dist(logit_p=logit_p), self.y_binary)
            weighted_log_likelihood = log_likelihood * self.weights
            pm.Potential('weighted_likelihood', weighted_log_likelihood.sum())
            
            self.final_trace = pm.sample(draws, tune=tune, cores=1, progressbar=False, return_inferencedata=False)

    def predict_proba(self, sample_data):
        """
        Predicts probabilities for sample data using both models.
        
        Args:
            sample_data: numpy array or pandas DataFrame of shape (n_samples, n_features)
            
        Returns:
            dict containing 'logistic_proba' and 'bayesian_proba' (probability of positive class)
        """
        if hasattr(sample_data, 'values'):
            sample_data = sample_data.values
            
        if not hasattr(self, 'final_logistic_model') or not hasattr(self, 'final_trace'):
            raise ValueError("Models have not been fitted. Call fit_final_models() first.")

        # Standard Logistic Regression Prediction
        # predict_proba returns [prob_0, prob_1], we want prob_1
        logistic_proba = self.final_logistic_model.predict_proba(sample_data)[:, 1]

        # Bayesian Logistic Regression Prediction
        alpha_samples = self.final_trace['alpha']
        beta_samples = self.final_trace['beta']
        
        probs_list = []
        for i in range(len(alpha_samples)):
            a = alpha_samples[i]
            b = beta_samples[i]
            logit = a + np.dot(sample_data, b)
            p_val = 1 / (1 + np.exp(-logit))
            probs_list.append(p_val)
        
        bayesian_proba = np.mean(probs_list, axis=0)
        
        return {
            'logistic_proba': logistic_proba,
            'bayesian_proba': bayesian_proba
        }
