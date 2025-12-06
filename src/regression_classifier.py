import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc, f1_score, roc_auc_score
import pymc as pm
import matplotlib.pyplot as plt
import os

def plot_regression_comparison(ridge_preds, bayesian_preds, true_labels, suffix="", plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)

    # --- Data for AD vs Rest ---
    y_true_ad = (true_labels == 1).astype(int)
    fpr_ridge_ad, tpr_ridge_ad, _ = roc_curve(y_true_ad, ridge_preds)
    roc_auc_ridge_ad = auc(fpr_ridge_ad, tpr_ridge_ad)
    fpr_bayes_ad, tpr_bayes_ad, _ = roc_curve(y_true_ad, bayesian_preds)
    roc_auc_bayes_ad = auc(fpr_bayes_ad, tpr_bayes_ad)

    # --- Data for Normal vs Rest ---
    y_true_norm = (true_labels == 0).astype(int)
    fpr_ridge_norm, tpr_ridge_norm, _ = roc_curve(y_true_norm, 1 - ridge_preds)
    roc_auc_ridge_norm = auc(fpr_ridge_norm, tpr_ridge_norm)
    fpr_bayes_norm, tpr_bayes_norm, _ = roc_curve(y_true_norm, 1 - bayesian_preds)
    roc_auc_bayes_norm = auc(fpr_bayes_norm, tpr_bayes_norm)

    # --- Plot 1: Compare Models for AD vs. Rest ---
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_ridge_ad, tpr_ridge_ad, color='red', linestyle='-', lw=2, label=f'Ridge (AUC = {roc_auc_ridge_ad:.2f})')
    plt.plot(fpr_bayes_ad, tpr_bayes_ad, color='red', linestyle='--', lw=2, label=f'Bayesian (AUC = {roc_auc_bayes_ad:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.title(f'Model Comparison: ROC for AD vs. Rest{suffix}', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, f'roc_compare_models_ad{suffix}.png'))
    plt.close()

    # --- Plot 2: Compare Models for Normal vs. Rest ---
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_ridge_norm, tpr_ridge_norm, color='green', linestyle='-', lw=2, label=f'Ridge (AUC = {roc_auc_ridge_norm:.2f})')
    plt.plot(fpr_bayes_norm, tpr_bayes_norm, color='green', linestyle='--', lw=2, label=f'Bayesian (AUC = {roc_auc_bayes_norm:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.title(f'Model Comparison: ROC for Normal vs. Rest{suffix}', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, f'roc_compare_models_normal{suffix}.png'))
    plt.close()

    # --- Plot 3: Compare Scenarios for Ridge Model ---
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_ridge_ad, tpr_ridge_ad, color='red', lw=2, label=f'AD vs. Rest (AUC = {roc_auc_ridge_ad:.2f})')
    plt.plot(fpr_ridge_norm, tpr_ridge_norm, color='green', lw=2, label=f'Normal vs. Rest (AUC = {roc_auc_ridge_norm:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.title(f'Ridge Model: Scenario Comparison{suffix}', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, f'roc_compare_scenarios_ridge{suffix}.png'))
    plt.close()

    # --- Plot 4: Compare Scenarios for Bayesian Model ---
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_bayes_ad, tpr_bayes_ad, color='red', lw=2, label=f'AD vs. Rest (AUC = {roc_auc_bayes_ad:.2f})')
    plt.plot(fpr_bayes_norm, tpr_bayes_norm, color='green', lw=2, label=f'Normal vs. Rest (AUC = {roc_auc_bayes_norm:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.title(f'Bayesian Model: Scenario Comparison{suffix}', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, f'roc_compare_scenarios_bayesian{suffix}.png'))
    plt.close()
    
    print(f"Generated 4 comparison plots with suffix '{suffix}' in '{plot_dir}'")

class RegressionClassifier:
    def __init__(self, args, X, y, n_splits=10, random_state=42):
        self.args = args
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _calculate_classification_metrics(self, y_true_continuous, y_pred_continuous):
        y_true_ad = (y_true_continuous == 1).astype(int)
        f1_ad = f1_score(y_true_ad, y_pred_continuous >= 0.5, zero_division=0)
        try:
            auc_ad = roc_auc_score(y_true_ad, y_pred_continuous)
        except ValueError:
            auc_ad = np.nan

        y_true_normal = (y_true_continuous == 0).astype(int)
        f1_normal = f1_score(y_true_normal, y_pred_continuous < 0.5, zero_division=0)
        try:
            auc_normal = roc_auc_score(y_true_normal, 1 - y_pred_continuous)
        except ValueError:
            auc_normal = np.nan
            
        return f1_ad, auc_ad, f1_normal, auc_normal

    def cv_ridge_regression(self):
        mse_scores, r2_scores, f1_ad_scores, auc_ad_scores, f1_normal_scores, auc_normal_scores = [], [], [], [], [], []
        y_true_all, y_pred_all = np.array([]), np.array([]) 

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mse_scores.append(mean_squared_error(y_val, y_pred))
            r2_scores.append(r2_score(y_val, y_pred))
            
            f1_ad, auc_ad, f1_normal, auc_normal = self._calculate_classification_metrics(y_val, y_pred)
            f1_ad_scores.append(f1_ad)
            auc_ad_scores.append(auc_ad)
            f1_normal_scores.append(f1_normal)
            auc_normal_scores.append(auc_normal)
            
            y_true_all = np.append(y_true_all, y_val)
            y_pred_all = np.append(y_pred_all, y_pred)

        metrics = {
            'mse': np.mean(mse_scores), 
            'r2_score': np.mean(r2_scores), 
            'f1_score_ad': np.nanmean(f1_ad_scores), 
            'auc_score_ad': np.nanmean(auc_ad_scores),
            'f1_score_normal': np.nanmean(f1_normal_scores),
            'auc_score_normal': np.nanmean(auc_normal_scores)
        }
        return metrics, y_true_all, y_pred_all

    def cv_bayesian_regression(self, draws=1000, tune=1000):
        mse_scores, r2_scores, f1_ad_scores, auc_ad_scores, f1_normal_scores, auc_normal_scores = [], [], [], [], [], []
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
                trace = pm.sample(draws, tune=tune, cores=1, progressbar=False, return_inferencedata=True)

            alpha_samples = trace.posterior["alpha"].values.flatten()
            beta_samples = trace.posterior["beta"].values.reshape(-1, n_features)
            
            y_pred_samples = []
            for i in range(len(alpha_samples)):
                y_pred_samples.append(alpha_samples[i] + np.dot(X_val, beta_samples[i]))

            y_pred_bayesian = np.mean(y_pred_samples, axis=0)

            mse_scores.append(mean_squared_error(y_val, y_pred_bayesian))
            r2_scores.append(r2_score(y_val, y_pred_bayesian))
            
            f1_ad, auc_ad, f1_normal, auc_normal = self._calculate_classification_metrics(y_val, y_pred_bayesian)
            f1_ad_scores.append(f1_ad)
            auc_ad_scores.append(auc_ad)
            f1_normal_scores.append(f1_normal)
            auc_normal_scores.append(auc_normal)

            y_true_all = np.append(y_true_all, y_val)
            y_pred_proba_all = np.append(y_pred_proba_all, y_pred_bayesian)

        metrics = {
            'mse': np.mean(mse_scores), 
            'r2_score': np.mean(r2_scores), 
            'f1_score_ad': np.nanmean(f1_ad_scores), 
            'auc_score_ad': np.nanmean(auc_ad_scores),
            'f1_score_normal': np.nanmean(f1_normal_scores),
            'auc_score_normal': np.nanmean(auc_normal_scores)
        }
        return metrics, y_true_all, y_pred_proba_all

    def fit_and_predict_full(self, draws=1000, tune=1000):
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
            trace = pm.sample(draws, tune=tune, cores=1, progressbar=False, return_inferencedata=True)
        
        alpha_samples = trace.posterior["alpha"].values.flatten()
        beta_samples = trace.posterior["beta"].values.reshape(-1, n_features)
        
        y_pred_samples = []
        for i in range(len(alpha_samples)):
            y_pred_samples.append(alpha_samples[i] + np.dot(self.X, beta_samples[i]))
        y_pred_bayesian = np.mean(y_pred_samples, axis=0)

        return self.y, y_pred_ridge, y_pred_bayesian
