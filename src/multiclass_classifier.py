import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import pymc as pm
import matplotlib.pyplot as plt
import os


class MultiClassifier:
    def __init__(self,args, X, y, n_splits=5, random_state=42):
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
    
    def _plot_multiclass_roc_curves(self, y_true, y_pred_proba, model_name, class_labels, plot_dir="plots"):
        os.makedirs(plot_dir, exist_ok=True)
        
        all_fpr = []
        all_tpr = []
        all_auc_scores = []

        plt.figure(figsize=(10, 8))
        for i, class_label in enumerate(class_labels):
            y_true_binary = (y_true == i).astype(int)
            y_pred_proba_class = y_pred_proba[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba_class)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (Class {class_label}, area = {roc_auc:.2f})')
            all_auc_scores.append(roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_name} (Multiclass Classification)')
        plt.legend(loc="lower right")
        
        plot_path = os.path.join(plot_dir, f'auc_curve_{model_name}_multiclass.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved multiclass AUC curves for {model_name} to {plot_path}")
        
        return np.mean(all_auc_scores)

    def cv_logistic_regression(self):
        acc_scores, f1_scores = [], []
        y_true_all, y_pred_proba_all = np.array([]), np.empty((0, self.n_classes))

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y_multiclass[train_index], self.y_multiclass[val_index]

            # Using Logistic Regression for multiclass classification
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)

            acc_scores.append(accuracy_score(y_val, y_pred))
            f1_scores.append(f1_score(y_val, y_pred, average='weighted'))

            y_true_all = np.append(y_true_all, y_val) 
            y_pred_proba_all = np.vstack((y_pred_proba_all, y_pred_proba)) 

        macro_auc_score = self._plot_multiclass_roc_curves(y_true_all, y_pred_proba_all, "Logistic_Regression", self.class_labels) 
        return {'accuracy': np.mean(acc_scores), 'f1_score': np.mean(f1_scores), 'auc_score': macro_auc_score}

    def cv_bayesian_logistic_regression(self, draws=500, tune=500):
        acc_scores, f1_scores = [], []
        n_features = self.X.shape[1]
        n_classes = self.n_classes
        y_true_all, y_pred_proba_all = np.array([]), np.empty((0, self.n_classes))

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y_multiclass[train_index], self.y_multiclass[val_index]

            with pm.Model() as multiclass_model:
                # Priors for intercepts and coefficients
                # Shape: (n_features, n_classes) for beta, (n_classes,) for alpha
                # We use a softmax link function, so we need parameters for each class.
                # Often one class is fixed as reference (e.g. all zeros), but here we'll estimate all and rely on softmax normalization.
                
                alpha = pm.Normal("alpha", mu=0, sigma=1, shape=n_classes)
                beta = pm.Laplace("beta", mu=0, b=self.args.multi_b, shape=(n_features, n_classes))
                
                # Linear model
                # X_train: (n_samples, n_features)
                # beta: (n_features, n_classes)
                # alpha: (n_classes,)
                # mu: (n_samples, n_classes)
                mu = alpha + pm.math.dot(X_train, beta)
                
                # Softmax transformation
                p = pm.math.softmax(mu, axis=1)
                
                # Likelihood
                y_obs = pm.Categorical("y_obs", p=p, observed=y_train)
                
                # Inference
                trace = pm.sample(draws, tune=tune, cores=1, progressbar=False, return_inferencedata=False)

            # Posterior prediction
            alpha_samples = trace['alpha'] # (n_samples, n_classes)
            beta_samples = trace['beta']   # (n_samples, n_features, n_classes)
            
            # Compute probabilities for validation set
            # We average the probabilities across posterior samples
            
            # X_val: (n_val, n_features)
            # We need to compute softmax(alpha + X_val @ beta) for each sample in trace
            
            # Let's do it in a vectorized way or loop if memory is concern.
            # Loop is safer for understanding:
            
            probs_list = []
            for i in range(len(alpha_samples)):
                a = alpha_samples[i]
                b = beta_samples[i]
                logit = a + np.dot(X_val, b)
                # Softmax
                # exp_logit = np.exp(logit - np.max(logit, axis=1, keepdims=True)) # Stable softmax
                # p_val = exp_logit / np.sum(exp_logit, axis=1, keepdims=True)
                
                # Scipy softmax is convenient but let's stick to numpy
                e_x = np.exp(logit - np.max(logit, axis=1, keepdims=True))
                p_val = e_x / e_x.sum(axis=1, keepdims=True)
                
                probs_list.append(p_val)
            
            avg_probs = np.mean(probs_list, axis=0)
            y_pred_bayesian = np.argmax(avg_probs, axis=1)

            acc_scores.append(accuracy_score(y_val, y_pred_bayesian))
            f1_scores.append(f1_score(y_val, y_pred_bayesian, average='weighted'))
            
            y_true_all = np.append(y_true_all, y_val) 
            y_pred_proba_all = np.vstack((y_pred_proba_all, avg_probs)) 

        macro_auc_score = self._plot_multiclass_roc_curves(y_true_all, y_pred_proba_all, "Bayesian_Logistic_Regression", self.class_labels) 
        return {'accuracy': np.mean(acc_scores), 'f1_score': np.mean(f1_scores), 'auc_score': macro_auc_score}

    def fit_final_models(self, draws=500, tune=500):
        """
        Fits both standard and Bayesian logistic regression models on the entire dataset.
        """
        # Standard Logistic Regression
        self.final_logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        self.final_logistic_model.fit(self.X, self.y_multiclass)

        # Bayesian Logistic Regression
        n_features = self.X.shape[1]
        n_classes = self.n_classes
        
        with pm.Model() as self.final_bayesian_model:
            alpha = pm.Normal("alpha", mu=0, sigma=1, shape=n_classes)
            beta = pm.Laplace("beta", mu=0, b=self.args.multi_b, shape=(n_features, n_classes))
            
            mu = alpha + pm.math.dot(self.X, beta)
            p = pm.math.softmax(mu, axis=1)
            
            y_obs = pm.Categorical("y_obs", p=p, observed=self.y_multiclass)
            
            self.final_trace = pm.sample(draws, tune=tune, cores=1, progressbar=False, return_inferencedata=False)

    def predict_proba(self, sample_data):
        """
        Predicts probabilities for sample data using both models.
        
        Args:
            sample_data: numpy array or pandas DataFrame of shape (n_samples, n_features)
            
        Returns:
            dict containing 'logistic_proba' and 'bayesian_proba'
        """
        if hasattr(sample_data, 'values'):
            sample_data = sample_data.values
            
        if not hasattr(self, 'final_logistic_model') or not hasattr(self, 'final_trace'):
            raise ValueError("Models have not been fitted. Call fit_final_models() first.")

        # Standard Logistic Regression Prediction
        logistic_proba = self.final_logistic_model.predict_proba(sample_data)

        # Bayesian Logistic Regression Prediction
        alpha_samples = self.final_trace['alpha']
        beta_samples = self.final_trace['beta']
        
        probs_list = []
        for i in range(len(alpha_samples)):
            a = alpha_samples[i]
            b = beta_samples[i]
            logit = a + np.dot(sample_data, b)
            
            # Softmax
            e_x = np.exp(logit - np.max(logit, axis=1, keepdims=True))
            p_val = e_x / e_x.sum(axis=1, keepdims=True)
            
            probs_list.append(p_val)
        
        bayesian_proba = np.mean(probs_list, axis=0)
        
        return {
            'logistic_proba': logistic_proba,
            'bayesian_proba': bayesian_proba,
            'classes': self.class_labels
        }
