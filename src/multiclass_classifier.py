import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
import pymc as pm
import matplotlib.pyplot as plt
import os

def plot_multiclass_comparison(true_labels, preds_dict, class_labels, suffix="", plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)
    n_classes = len(class_labels)
    
    y_true_bin = label_binarize(true_labels, classes=range(n_classes))
    
    color_map = {'Alzheimer': 'red', 'MCI': 'orange', 'Normal': 'green'}
    colors = [color_map.get(label, 'gray') for label in class_labels]

    model_name_logistic = "Logistic Model"
    y_pred_proba_logistic = preds_dict['Logistic']
    
    fpr_logistic = dict()
    tpr_logistic = dict()
    roc_auc_logistic = dict()

    for i in range(n_classes):
        fpr_logistic[i], tpr_logistic[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba_logistic[:, i])
        roc_auc_logistic[i] = auc(fpr_logistic[i], tpr_logistic[i])

    fpr_logistic["micro"], tpr_logistic["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba_logistic.ravel())
    roc_auc_logistic["micro"] = auc(fpr_logistic["micro"], tpr_logistic["micro"])

    all_fpr_logistic = np.unique(np.concatenate([fpr_logistic[i] for i in range(n_classes)]))
    mean_tpr_logistic = np.zeros_like(all_fpr_logistic)
    for i in range(n_classes):
        mean_tpr_logistic += np.interp(all_fpr_logistic, fpr_logistic[i], tpr_logistic[i])
    mean_tpr_logistic /= n_classes
    fpr_logistic["macro"] = all_fpr_logistic
    tpr_logistic["macro"] = mean_tpr_logistic
    roc_auc_logistic["macro"] = auc(fpr_logistic["macro"], tpr_logistic["macro"])

    plt.figure(figsize=(11, 9))
    plt.plot(fpr_logistic["micro"], tpr_logistic["micro"],
            label=f'Micro-average ROC (area = {roc_auc_logistic["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=3)
    plt.plot(fpr_logistic["macro"], tpr_logistic["macro"],
            label=f'Macro-average ROC (area = {roc_auc_logistic["macro"]:.2f})',
            color='navy', linestyle=':', linewidth=3)
    for i, color in enumerate(colors):
        plt.plot(fpr_logistic[i], tpr_logistic[i], color=color, lw=2,
                 label=f'ROC of class {class_labels[i]} (area = {roc_auc_logistic[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name_logistic}: One-vs-Rest ROC by Class{suffix}', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 8})
    plot_path_logistic = os.path.join(plot_dir, f'roc_multiclass_logistic{suffix}.png')
    plt.savefig(plot_path_logistic)
    plt.close()
    print(f"Saved multiclass ROC for {model_name_logistic} to {plot_path_logistic}")

    model_name_bayesian = "Bayesian Model"
    y_pred_proba_bayesian = preds_dict['Bayesian']

    fpr_bayesian = dict()
    tpr_bayesian = dict()
    roc_auc_bayesian = dict()

    for i in range(n_classes):
        fpr_bayesian[i], tpr_bayesian[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba_bayesian[:, i])
        roc_auc_bayesian[i] = auc(fpr_bayesian[i], tpr_bayesian[i])

    fpr_bayesian["micro"], tpr_bayesian["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba_bayesian.ravel())
    roc_auc_bayesian["micro"] = auc(fpr_bayesian["micro"], tpr_bayesian["micro"])

    all_fpr_bayesian = np.unique(np.concatenate([fpr_bayesian[i] for i in range(n_classes)]))
    mean_tpr_bayesian = np.zeros_like(all_fpr_bayesian)
    for i in range(n_classes):
        mean_tpr_bayesian += np.interp(all_fpr_bayesian, fpr_bayesian[i], tpr_bayesian[i])
    mean_tpr_bayesian /= n_classes
    fpr_bayesian["macro"] = all_fpr_bayesian
    tpr_bayesian["macro"] = mean_tpr_bayesian
    roc_auc_bayesian["macro"] = auc(fpr_bayesian["macro"], tpr_bayesian["macro"])

    plt.figure(figsize=(11, 9))
    plt.plot(fpr_bayesian["micro"], tpr_bayesian["micro"],
            label=f'Micro-average ROC (area = {roc_auc_bayesian["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=3)
    plt.plot(fpr_bayesian["macro"], tpr_bayesian["macro"],
            label=f'Macro-average ROC (area = {roc_auc_bayesian["macro"]:.2f})',
            color='navy', linestyle=':', linewidth=3)
    for i, color in enumerate(colors):
        plt.plot(fpr_bayesian[i], tpr_bayesian[i], color=color, lw=2,
                 label=f'ROC of class {class_labels[i]} (area = {roc_auc_bayesian[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name_bayesian}: One-vs-Rest ROC by Class{suffix}', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 8})
    plot_path_bayesian = os.path.join(plot_dir, f'roc_multiclass_bayesian{suffix}.png')
    plt.savefig(plot_path_bayesian)
    plt.close()
    print(f"Saved multiclass ROC for {model_name_bayesian} to {plot_path_bayesian}")

class MultiClassifier:
    def __init__(self,args, X, y, n_splits=10, random_state=42):
        self.args = args
        self.X = X
        self.y_original = y
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.y_multiclass, self.class_labels = self._prepare_target()
        self.n_classes = len(self.class_labels)

    def _prepare_target(self):
        y_str = self.y_original.apply(lambda val: 'Normal' if str(val).startswith('C') else ('Alzheimer' if str(val).startswith('AD') else 'MCI'))
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_str)
        return y_encoded, encoder.classes_

    def _get_macro_auc(self, y_true, y_pred_proba):
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        return roc_auc_score(y_true_bin, y_pred_proba, average='macro')
    
    def cv_logistic_regression(self):
        acc_scores, f1_scores = [], []
        y_true_all, y_pred_proba_all = np.array([]), np.empty((0, self.n_classes))

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y_multiclass[train_index], self.y_multiclass[val_index]

            model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)

            acc_scores.append(accuracy_score(y_val, y_pred))
            f1_scores.append(f1_score(y_val, y_pred, average='weighted'))

            y_true_all = np.append(y_true_all, y_val) 
            y_pred_proba_all = np.vstack((y_pred_proba_all, y_pred_proba)) 

        macro_auc_score = self._get_macro_auc(y_true_all, y_pred_proba_all)
        metrics = {'accuracy': np.mean(acc_scores), 'f1_score': np.mean(f1_scores), 'auc_score': macro_auc_score}
        return metrics, y_true_all, y_pred_proba_all

    def cv_bayesian_logistic_regression(self, draws=500, tune=500):
        acc_scores, f1_scores = [], []
        n_features = self.X.shape[1]
        y_true_all, y_pred_proba_all = np.array([]), np.empty((0, self.n_classes))

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y_multiclass[train_index], self.y_multiclass[val_index]

            with pm.Model() as multiclass_model:
                alpha = pm.Normal("alpha", mu=0, sigma=1, shape=self.n_classes)
                beta = pm.Laplace("beta", mu=0, b=self.args.multi_b, shape=(n_features, self.n_classes))
                mu = alpha + pm.math.dot(X_train, beta)
                p = pm.math.softmax(mu, axis=1)
                y_obs = pm.Categorical("y_obs", p=p, observed=y_train)
                trace = pm.sample(draws, tune=tune, cores=1, progressbar=False, return_inferencedata=True)

            alpha_samples = trace.posterior['alpha'].values.reshape(-1, self.n_classes)
            beta_samples = trace.posterior['beta'].values.reshape(-1, n_features, self.n_classes)
            
            probs_list = []
            for i in range(len(alpha_samples)):
                a = alpha_samples[i]
                b = beta_samples[i]
                logit = a + np.dot(X_val, b)
                e_x = np.exp(logit - np.max(logit, axis=1, keepdims=True))
                p_val = e_x / e_x.sum(axis=1, keepdims=True)
                probs_list.append(p_val)
            
            avg_probs = np.mean(probs_list, axis=0)
            y_pred_bayesian = np.argmax(avg_probs, axis=1)

            acc_scores.append(accuracy_score(y_val, y_pred_bayesian))
            f1_scores.append(f1_score(y_val, y_pred_bayesian, average='weighted'))
            
            y_true_all = np.append(y_true_all, y_val) 
            y_pred_proba_all = np.vstack((y_pred_proba_all, avg_probs)) 

        macro_auc_score = self._get_macro_auc(y_true_all, y_pred_proba_all)
        metrics = {'accuracy': np.mean(acc_scores), 'f1_score': np.mean(f1_scores), 'auc_score': macro_auc_score}
        return metrics, y_true_all, y_pred_proba_all

    def fit_and_predict_full(self, draws=500, tune=500):
        logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        logistic_model.fit(self.X, self.y_multiclass)
        logistic_preds = logistic_model.predict_proba(self.X)

        n_features = self.X.shape[1]
        with pm.Model() as final_bayesian_model:
            alpha = pm.Normal("alpha", mu=0, sigma=1, shape=self.n_classes)
            beta = pm.Laplace("beta", mu=0, b=self.args.multi_b, shape=(n_features, self.n_classes))
            mu = alpha + pm.math.dot(self.X, beta)
            p = pm.math.softmax(mu, axis=1)
            y_obs = pm.Categorical("y_obs", p=p, observed=self.y_multiclass)
            trace = pm.sample(draws, tune=tune, cores=1, progressbar=False, return_inferencedata=True)
        
        alpha_samples = trace.posterior['alpha'].values.reshape(-1, self.n_classes)
        beta_samples = trace.posterior['beta'].values.reshape(-1, n_features, self.n_classes)
        
        probs_list = []
        for i in range(len(alpha_samples)):
            a = alpha_samples[i]
            b = beta_samples[i]
            logit = a + np.dot(self.X, b)
            e_x = np.exp(logit - np.max(logit, axis=1, keepdims=True))
            p_val = e_x / e_x.sum(axis=1, keepdims=True)
            probs_list.append(p_val)
        
        bayesian_preds = np.mean(probs_list, axis=0)
        
        return self.y_multiclass, logistic_preds, bayesian_preds
