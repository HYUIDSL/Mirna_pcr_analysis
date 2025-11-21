import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

class MultiClassifier:
    def __init__(self, X, y, n_splits=5, random_state=42):
        self.X = X
        self.y_original = y
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.y_multiclass, self.class_labels = self._prepare_target()

    def _prepare_target(self):
        y_str = self.y_original.apply(lambda val: 'Normal' if str(val).startswith('C') else ('Alzheimer' if str(val).startswith('AD') else 'MCI'))
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_str)
        return y_encoded, encoder.classes_

    def run_multiclass_classification(self):
        acc_scores, f1_scores = [], []

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y_multiclass[train_index], self.y_multiclass[val_index]

            # Using Logistic Regression for multiclass classification
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            acc_scores.append(accuracy_score(y_val, y_pred))
            f1_scores.append(f1_score(y_val, y_pred, average='weighted'))

        return {'accuracy': np.mean(acc_scores), 'f1_score': np.mean(f1_scores)}
