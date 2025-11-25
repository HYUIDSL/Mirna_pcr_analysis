import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pymc as pm
from imblearn.over_sampling import SMOTE


class MultiClassifier:
    def __init__(self,args, X, y, n_splits=5, random_state=42):
        self.args = args
        # The model now expects unscaled data
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

    def cv_logistic_regression(self):
        acc_scores, f1_scores = [], []
        total_cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        all_y_true = []
        all_y_proba = []

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y_multiclass[train_index], self.y_multiclass[val_index]

            # 1. Scale data inside the CV loop
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 2. Apply SMOTE to the scaled training data
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

            # 3. Fit model with the tuned hyperparameter
            model = LogisticRegression(C=0.1061, solver='lbfgs', max_iter=1000)
            model.fit(X_train_smote, y_train_smote)
            
            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled)

            acc_scores.append(accuracy_score(y_val, y_pred))
            f1_scores.append(f1_score(y_val, y_pred, average='macro', zero_division=0))
            total_cm += confusion_matrix(y_val, y_pred, labels=range(self.n_classes))
            all_y_true.extend(y_val)
            all_y_proba.append(y_proba)

        all_y_proba = np.concatenate(all_y_proba, axis=0)
        return {
            'accuracy': np.mean(acc_scores), 
            'f1_score': np.mean(f1_scores), 
            'confusion_matrix': total_cm,
            'y_true': np.array(all_y_true),
            'y_proba': all_y_proba
        }

    def cv_bayesian_logistic_regression(self, draws=1000, tune=1000):
        acc_scores, f1_scores = [], []
        total_cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        all_y_true = []
        all_y_proba = []
        n_features = self.X.shape[1]
        n_classes = self.n_classes

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y_multiclass[train_index], self.y_multiclass[val_index]

            # 1. Scale data inside the CV loop
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 2. Apply SMOTE to the scaled training data
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

            with pm.Model() as multiclass_model:
                alpha = pm.Normal("alpha", mu=0, sigma=1, shape=n_classes)
                beta = pm.Laplace("beta", mu=0, b=self.args.multi_b, shape=(n_features, n_classes))
                mu = alpha + pm.math.dot(X_train_smote, beta)
                p = pm.math.softmax(mu, axis=1)
                y_obs = pm.Categorical("y_obs", p=p, observed=y_train_smote)
                trace = pm.sample(draws, tune=tune, chains=4, cores=4, progressbar=False, return_inferencedata=False)

            # Posterior prediction
            alpha_samples = trace['alpha']
            beta_samples = trace['beta']
            
            probs_list = []
            for i in range(len(alpha_samples)):
                a = alpha_samples[i]
                b = beta_samples[i]
                logit = a + np.dot(X_val_scaled, b)
                e_x = np.exp(logit - np.max(logit, axis=1, keepdims=True))
                p_val = e_x / e_x.sum(axis=1, keepdims=True)
                probs_list.append(p_val)
            
            avg_probs = np.mean(probs_list, axis=0)
            y_pred_bayesian = np.argmax(avg_probs, axis=1)

            acc_scores.append(accuracy_score(y_val, y_pred_bayesian))
            f1_scores.append(f1_score(y_val, y_pred_bayesian, average='macro', zero_division=0))
            total_cm += confusion_matrix(y_val, y_pred_bayesian, labels=range(self.n_classes))
            all_y_true.extend(y_val)
            all_y_proba.append(avg_probs)

        all_y_proba = np.concatenate(all_y_proba, axis=0)
        return {
            'accuracy': np.mean(acc_scores), 
            'f1_score': np.mean(f1_scores), 
            'confusion_matrix': total_cm,
            'y_true': np.array(all_y_true),
            'y_proba': all_y_proba
        }

    def fit_final_models(self, draws=1000, tune=1000):
        """
        Fits both standard and Bayesian logistic regression models on the entire dataset.
        """
        # 1. Scale the entire dataset for final model training
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # 2. Apply SMOTE to the scaled data
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_scaled, self.y_multiclass)

        # 3. Fit models on the balanced and scaled data
        # Standard Logistic Regression
        self.final_logistic_model = LogisticRegression(C=0.1061, solver='lbfgs', max_iter=1000)
        self.final_logistic_model.fit(X_smote, y_smote)

        # Bayesian Logistic Regression
        n_features = self.X.shape[1]
        n_classes = self.n_classes
        
        with pm.Model() as self.final_bayesian_model:
            alpha = pm.Normal("alpha", mu=0, sigma=1, shape=n_classes)
            beta = pm.Laplace("beta", mu=0, b=self.args.multi_b, shape=(n_features, n_classes))
            
            mu = alpha + pm.math.dot(X_smote, beta)
            p = pm.math.softmax(mu, axis=1)
            
            y_obs = pm.Categorical("y_obs", p=p, observed=y_smote)
            
            self.final_trace = pm.sample(draws, tune=tune, chains=4, cores=4, progressbar=False, return_inferencedata=False)

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
            
            e_x = np.exp(logit - np.max(logit, axis=1, keepdims=True))
            p_val = e_x / e_x.sum(axis=1, keepdims=True)
            
            probs_list.append(p_val)
        
        bayesian_proba = np.mean(probs_list, axis=0)
        
        return {
            'logistic_proba': logistic_proba,
            'bayesian_proba': bayesian_proba,
            'classes': self.class_labels
        }
