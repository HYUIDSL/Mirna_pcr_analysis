import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

class HyperparameterTuner:
    """
    A class to tune hyperparameters for classification models using Optuna.
    """
    def __init__(self, X, y, n_splits=5, random_state=42):
        self.X = X
        self.y = y
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _objective_logistic(self, trial):
        """
        Objective function for tuning Logistic Regression.
        """
        # Define the hyperparameter search space
        C = trial.suggest_float('C', 1e-4, 1e3, log=True)
        
        f1_scores = []
        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]

            # Pipeline: Scale -> SMOTE -> Fit
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

            model = LogisticRegression(C=C, solver='lbfgs', max_iter=1000)
            model.fit(X_train_smote, y_train_smote)
            
            y_pred = model.predict(X_val_scaled)
            f1_scores.append(f1_score(y_val, y_pred, average='macro', zero_division=0))

        # Optuna maximizes the return value, so we return the mean F1 score directly.
        return np.mean(f1_scores)

    def tune_logistic(self, n_trials=1000):
        """
        Run the hyperparameter tuning study for Logistic Regression.
        """
        # We want to maximize the F1 score, so we set the direction to 'maximize'.
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective_logistic, n_trials=n_trials)

        print("\n--- Logistic Regression Tuning Results ---")
        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value (F1 Score): {trial.value:.4f}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        return study.best_params

# Note: Bayesian model tuning is more complex and will be added later if needed.
# The PyMC sampling process within an Optuna loop can be very time-consuming.
