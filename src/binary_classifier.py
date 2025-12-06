import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import KFold
import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_binary_comparison(true_labels, preds_dict, suffix="", plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))

    styles = [{'color': 'cornflowerblue', 'linestyle': '-'}, {'color': 'red', 'linestyle': '--'}]
    
    for (model_name, y_pred_proba), style in zip(preds_dict.items(), styles):
        fpr, tpr, _ = roc_curve(true_labels, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=style['color'], linestyle=style['linestyle'], lw=2, 
                 label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Model Comparison: ROC for Alzheimers vs. Others{suffix}', fontsize=14)
    plt.legend(loc="lower right")
    
    plot_path = os.path.join(plot_dir, f'roc_compare_models_binary{suffix}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved binary comparison plot to {plot_path}")

class BinaryClassifier:
    def __init__(self,args, X, y, feature_names, n_splits=10, random_state=42):
        self.args = args
        self.X = X
        self.y_original = y
        self.feature_names = feature_names
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.y_binary, self.weights = self._prepare_target()

    def _prepare_target(self):
        y_str = self.y_original.apply(lambda val: 'Normal' if str(val).startswith('C') else ('Alzheimer' if str(val).startswith('AD') else 'MCI'))
        y_binary = y_str.apply(lambda val: 1 if val == 'Alzheimer' else 0)
        weights = np.ones(len(y_str))
        mci_weight = 0.8
        weights[y_str[y_str == 'MCI'].index] = mci_weight
        return y_binary, weights

    def cv_logistic_regression(self):
        acc_scores, recall_scores, f1_scores, auc_scores = [], [], [], []
        y_true_all, y_pred_proba_all = np.array([]), np.array([]) 

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y_binary.iloc[train_index], self.y_binary.iloc[val_index]
            weights_train, weights_val = self.weights[train_index], self.weights[val_index]

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1] 

            acc_scores.append(accuracy_score(y_val, y_pred, sample_weight=weights_val))
            recall_scores.append(recall_score(y_val, y_pred, sample_weight=weights_val, zero_division=0))
            f1_scores.append(f1_score(y_val, y_pred, sample_weight=weights_val, zero_division=0))
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            auc_scores.append(auc(fpr, tpr))

            y_true_all = np.append(y_true_all, y_val) 
            y_pred_proba_all = np.append(y_pred_proba_all, y_pred_proba)
        
        metrics = {'accuracy': np.mean(acc_scores), 'recall': np.mean(recall_scores), 'f1_score': np.mean(f1_scores), 'auc_score': np.mean(auc_scores)}
        return metrics, y_true_all, y_pred_proba_all

    def cv_bayesian_logistic_regression(self):
        acc_scores, recall_scores, f1_scores, auc_scores = [], [], [], []
        n_features = self.X.shape[1]
        y_true_all, y_pred_proba_all = np.array([]), np.array([]) 

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
                trace = pm.sample(500, tune=500, cores=1, progressbar=False, return_inferencedata=True)

            alpha_samples = trace.posterior["alpha"].values.flatten()
            beta_samples = trace.posterior["beta"].values.reshape(-1, n_features)
            
            all_probs = []
            for i in range(len(alpha_samples)):
                alpha_s = alpha_samples[i]
                beta_s = beta_samples[i]
                all_probs.append(1 / (1 + np.exp(-(alpha_s + np.dot(X_val, beta_s)))))

            y_prob_bayesian = np.mean(all_probs, axis=0)
            y_pred_bayesian = (y_prob_bayesian > 0.5).astype(int)

            acc_scores.append(accuracy_score(y_val, y_pred_bayesian, sample_weight=weights_val))
            recall_scores.append(recall_score(y_val, y_pred_bayesian, sample_weight=weights_val, zero_division=0))
            f1_scores.append(f1_score(y_val, y_pred_bayesian, sample_weight=weights_val, zero_division=0))
            fpr, tpr, _ = roc_curve(y_val, y_prob_bayesian)
            auc_scores.append(auc(fpr, tpr))
            
            y_true_all = np.append(y_true_all, y_val) 
            y_pred_proba_all = np.append(y_pred_proba_all, y_prob_bayesian) 

        metrics = {'accuracy': np.mean(acc_scores), 'recall': np.mean(recall_scores), 'f1_score': np.mean(f1_scores), 'auc_score': np.mean(auc_scores)}
        return metrics, y_true_all, y_pred_proba_all

    def fit_and_predict_full(self, draws=500, tune=500):
        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(self.X, self.y_binary, sample_weight=self.weights)
        logistic_preds = logistic_model.predict_proba(self.X)[:, 1]

        n_features = self.X.shape[1]
        with pm.Model() as final_bayesian_model:
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            beta = pm.Laplace("beta", mu=0, b=self.args.binary_b, shape=n_features)
            logit_p = alpha + pm.math.dot(self.X, beta)
            log_likelihood = pm.logp(pm.Bernoulli.dist(logit_p=logit_p), self.y_binary)
            weighted_log_likelihood = log_likelihood * self.weights
            pm.Potential('weighted_likelihood', weighted_log_likelihood.sum())
            trace = pm.sample(draws, tune=tune, cores=1, progressbar=False, return_inferencedata=True)
        
        alpha_samples = trace.posterior["alpha"].values.flatten()
        beta_samples = trace.posterior["beta"].values.reshape(-1, n_features)
        
        all_probs = []
        for i in range(len(alpha_samples)):
            all_probs.append(alpha_samples[i] + np.dot(self.X, beta_samples[i]))
        
        y_prob_bayesian = np.mean([1 / (1 + np.exp(-p)) for p in all_probs], axis=0)
        
        return self.y_binary, logistic_preds, y_prob_bayesian
