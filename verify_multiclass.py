import pandas as pd
from src.preprocess import Preprocessor
from src.multiclass_classifier import MultiClassifier

def verify():
    print("--- Starting Verification ---")
    # Load data
    preprocessor = Preprocessor('mirna_v3.xlsx')
    X_combined, y_original = preprocessor.preprocess()
    X_scaled = preprocessor.get_scaled_data()
    
    print("Data loaded and preprocessed.")
    print(f"X shape: {X_scaled.shape}")
    print(f"y shape: {y_original.shape}")

    # Initialize MultiClassifier
    multi_classifier = MultiClassifier(X_scaled, y_original, n_splits=2) # Reduced splits for quick verification
    
    # Run Standard Logistic Regression
    print("\nRunning Standard Logistic Regression...")
    logistic_results = multi_classifier.run_logistic_regression()
    print("Standard Logistic Regression Results:", logistic_results)

    # Run Bayesian Logistic Regression
    print("\nRunning Bayesian Logistic Regression...")
    bayesian_results = multi_classifier.run_bayesian_logistic_regression(draws=50, tune=50)
    print("Bayesian Logistic Regression Results:", bayesian_results)

    # Verify Prediction Logic
    print("\n--- Verifying Prediction Logic ---")
    print("Fitting final models on full dataset...")
    multi_classifier.fit_final_models(draws=50, tune=50)
    
    # Create a dummy sample (using the first row of scaled data)
    sample_data = X_scaled[0].reshape(1, -1)
    print(f"Predicting for sample data shape: {sample_data.shape}")
    
    prediction_results = multi_classifier.predict_proba(sample_data)
    print("Prediction Results:")
    print("Classes:", prediction_results['classes'])
    print("Logistic Probabilities:", prediction_results['logistic_proba'])
    print("Bayesian Probabilities:", prediction_results['bayesian_proba'])

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify()
