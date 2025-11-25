import pandas as pd
from src.preprocess import Preprocessor
from src.binary_classifier import BinaryClassifier

def verify():
    print("--- Starting Binary Verification ---")
    # Load data
    preprocessor = Preprocessor('mirna_v3.xlsx')
    X_combined, y_original = preprocessor.preprocess()
    X_scaled = preprocessor.get_scaled_data()
    
    print("Data loaded and preprocessed.")
    print(f"X shape: {X_scaled.shape}")
    print(f"y shape: {y_original.shape}")

    # Initialize BinaryClassifier
    binary_classifier = BinaryClassifier(X_scaled, y_original, n_splits=2)
    
    # Verify Prediction Logic
    print("\n--- Verifying Prediction Logic ---")
    print("Fitting final models on full dataset...")
    binary_classifier.fit_final_models(draws=50, tune=50)
    
    # Create a dummy sample (using the first row of scaled data)
    sample_data = X_scaled[0].reshape(1, -1)
    print(f"Predicting for sample data shape: {sample_data.shape}")
    
    prediction_results = binary_classifier.predict_proba(sample_data)
    print("Prediction Results:")
    print("Logistic Probability (Positive Class):", prediction_results['logistic_proba'])
    print("Bayesian Probability (Positive Class):", prediction_results['bayesian_proba'])

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify()
