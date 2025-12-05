import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc, f1_score, roc_auc_score
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os

class RegressionClassifier:
    def __init__(self, args, X, y, n_splits=5, random_state=42):
        self.args = args
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _calculate_classification_metrics(self, y_true_continuous, y_pred_continuous):
        y_true_binary = (y_true_continuous >= 0.5).astype(int)
        
        y_pred_binary = (y_pred_continuous >= 0.5).astype(int)
        
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        try:
            auc_score = roc_auc_score(y_true_binary, y_pred_continuous)
        except ValueError:
            auc_score = np.nan 
            
        return f1, auc_score

    def cv_ridge_regression(self):
        mse_scores, r2_scores, f1_scores, auc_scores = [], [], [], []
        y_true_all, y_pred_all = np.array([]), np.array([]) 

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

            model = Ridge(alpha=1.0) # tune
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mse_scores.append(mean_squared_error(y_val, y_pred))
            r2_scores.append(r2_score(y_val, y_pred))
            
            f1, auc_score = self._calculate_classification_metrics(y_val, y_pred)
            f1_scores.append(f1)
            auc_scores.append(auc_score)
            
            y_true_all = np.append(y_true_all, y_val)
            y_pred_all = np.append(y_pred_all, y_pred)

        y_true_for_plot = (y_true_all >= 0.5).astype(int)
        self.plot_auc_like_curve(y_true_for_plot, y_pred_all, "Ridge_Regression")

        return {'mse': np.mean(mse_scores), 'r2_score': np.mean(r2_scores), 
                'f1_score': np.nanmean(f1_scores), 'auc_score': np.nanmean(auc_scores)}

    def cv_bayesian_regression(self, draws=500, tune=500):
        mse_scores, r2_scores, f1_scores, auc_scores = [], [], [], []
        n_features = self.X.shape[1]
        y_true_all, y_pred_proba_all = np.array([]), np.array([]) 

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

            with pm.Model() as bayesian_model:
                alpha = pm.Normal("alpha", mu=0, sigma=10)
                beta = pm.Normal("beta", mu=0, sigma=self.args.regression_b, shape=n_features)
                sigma = pm.HalfNormal("sigma", sigma=1) 

                mu = alpha + pm.math.dot(X_train, beta)

                y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

                trace = pm.sample(draws, tune=tune, cores=1, progressbar=False, return_inferencedata=False)

            alpha_samples = trace['alpha']
            beta_samples = trace['beta']
            sigma_samples = trace['sigma']

            y_pred_samples = []
            for i in range(len(alpha_samples)):
                y_pred_samples.append(alpha_samples[i] + np.dot(X_val, beta_samples[i]))

            y_pred_bayesian = np.mean(y_pred_samples, axis=0) 

            mse_scores.append(mean_squared_error(y_val, y_pred_bayesian))
            r2_scores.append(r2_score(y_val, y_pred_bayesian))
            
            f1, auc_score = self._calculate_classification_metrics(y_val, y_pred_bayesian)
            f1_scores.append(f1)
            auc_scores.append(auc_score)
            
            y_true_all = np.append(y_true_all, y_val)
            y_pred_proba_all = np.append(y_pred_proba_all, y_pred_bayesian)

        y_true_for_plot = (y_true_all >= 0.5).astype(int)
        self.plot_auc_like_curve(y_true_for_plot, y_pred_proba_all, "Bayesian_Regression")

        return {'mse': np.mean(mse_scores), 'r2_score': np.mean(r2_scores), 
                'f1_score': np.nanmean(f1_scores), 'auc_score': np.nanmean(auc_scores)}

    def plot_auc_like_curve(self, y_true_binary, y_pred_proba, model_name, plot_dir="plots"):
        os.makedirs(plot_dir, exist_ok=True)

        y_true_binary = (y_true_binary > 0.5).astype(int) 

        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_name} (Regression Mode)')
        plt.legend(loc="lower right")
        
        plot_path = os.path.join(plot_dir, f'auc_curve_{model_name}_regression.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved AUC-like curve for {model_name} to {plot_path}")

    def fit_and_plot(self, plot_dir="plots"):
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(self.X, self.y)
        y_pred_ridge = ridge_model.predict(self.X)

        n_features = self.X.shape[1]
        with pm.Model() as bayesian_model:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=self.args.regression_b, shape=n_features)
            sigma = pm.HalfNormal("sigma", sigma=1)
            mu = alpha + pm.math.dot(self.X, beta)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.y)
            trace = pm.sample(500, tune=500, cores=1, progressbar=False, return_inferencedata=False)
        
        alpha_samples = trace['alpha']
        beta_samples = trace['beta']
        y_pred_samples = []
        for i in range(len(alpha_samples)):
            y_pred_samples.append(alpha_samples[i] + np.dot(self.X, beta_samples[i]))
        y_pred_bayesian = np.mean(y_pred_samples, axis=0)

        y_true_for_auc = (self.y >= 0.5).astype(int)

        self.plot_auc_like_curve(y_true_for_auc, y_pred_ridge, "Ridge_Regression", plot_dir)
        self.plot_auc_like_curve(y_true_for_auc, y_pred_bayesian, "Bayesian_Regression", plot_dir)
