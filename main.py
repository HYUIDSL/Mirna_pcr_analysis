import argparse
import pandas as pd
import numpy as np
from src.preprocess import Preprocessor
from src.binary_classifier import BinaryClassifier
from src.multiclass_classifier import MultiClassifier
from src.visualize import plot_confusion_matrix_heatmap, plot_roc_curves
from src.feature_engineering import FeatureEngineer
from src.tuner import HyperparameterTuner

def main():
    # --- 0. 인자 파싱 설정 (옵션 선택 기능) ---
    parser = argparse.ArgumentParser(description='Run Classification Model')
    parser.add_argument('--mode', type=str, choices=['binary', 'multi'], default='multi',
                        help='Choose classification mode: "binary" or "multi"')
    parser.add_argument('--tune', action='store_true',
                        help='If set, run hyperparameter tuning for the selected mode.')
    parser.add_argument('--binary_b', type=float, default=0.5,
                        help='Hyperparameter b for binary Bayesian logistic regression')
    parser.add_argument('--multi_b', type=float, default=0.1,
                        help='Hyperparameter b for multiclass Bayesian logistic regression')
    args = parser.parse_args()

    # --- 1. 데이터 로딩 및 전처리 (공통) ---
    print("--- Starting Preprocessing ---")
    preprocessor = Preprocessor('mirna_v3.xlsx')
    X_combined, y_original = preprocessor.preprocess()
    
    # --- 1.5 피처 엔지니어링 ---
    feature_engineer = FeatureEngineer(X_combined)
    X_engineered = feature_engineer.transform()

    print("--- Preprocessing and Feature Engineering Complete ---")
    
    # --- 2. 모드에 따른 분류 모델 실행 ---
    if args.tune:
        # --- Hyperparameter Tuning Logic ---
        print(f"\n--- Starting Hyperparameter Tuning for {args.mode} mode ---")
        if args.mode == 'multi':
            # Prepare multiclass target variable
            y_str = y_original.apply(lambda val: 'Normal' if str(val).startswith('C') else ('Alzheimer' if str(val).startswith('AD') else 'MCI'))
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            y_multi = encoder.fit_transform(y_str)
            
            tuner = HyperparameterTuner(X_engineered, y_multi)
            best_params = tuner.tune_logistic(n_trials=100)
            print("\nTuning complete. Best parameters for Logistic Regression:", best_params)
        else:
            # TODO: Implement tuning for binary mode if needed
            print("Tuning for binary mode is not yet implemented.")

    else:
        # --- Standard Model Evaluation Logic ---
        if args.mode == 'binary':
            # === Binary Classification Logic ===
            print("\n--- Starting Binary Classification ---")
            binary_classifier = BinaryClassifier(args, X_engineered, y_original)
            
            print("Running standard logistic regression...")
            logistic_results = binary_classifier.cv_logistic_regression()
            
            print("Running Bayesian logistic regression...")
            bayesian_results = binary_classifier.cv_bayesian_logistic_regression()
            print("--- Binary Classification Complete ---")

            # 결과 요약
            print("\n" + "="*60)
            print("        Binary Cross-Validation Results (Average)")
            print("="*60)

            results_summary = pd.DataFrame({
                'Metric': ['Accuracy', 'F1 Score'],
                'Logistic': [logistic_results['accuracy'], logistic_results['f1_score']],
                'Bayesian': [bayesian_results['accuracy'], bayesian_results['f1_score']]
            })
            print(results_summary.set_index('Metric').round(4))

        elif args.mode == 'multi':
            # === Multiclass Classification Logic ===
            print("\n--- Starting Multiclass Classification ---")
            multi_classifier = MultiClassifier(args, X_engineered, y_original)
            
            print("Running standard logistic regression (Multiclass)...")
            multi_logistic_results = multi_classifier.cv_logistic_regression()
            
            print("Running Bayesian logistic regression (Multiclass)...")
            multi_bayesian_results = multi_classifier.cv_bayesian_logistic_regression()
            
            print("--- Multiclass Classification Complete ---")

            # 결과 요약
            print("\n" + "="*60)
            print("        Multiclass Cross-Validation Results (Average)")
            print("="*60)
            
            multi_results_summary = pd.DataFrame({
                'Metric': ['Accuracy', 'F1 Score (Macro)'],
                'Logistic': [multi_logistic_results['accuracy'], multi_logistic_results['f1_score']],
                'Bayesian': [multi_bayesian_results['accuracy'], multi_bayesian_results['f1_score']]
            })
            print(multi_results_summary.set_index('Metric').round(4))
        
            print("\n" + "-"*60)
            print("Confusion Matrix Details")
            print("-" * 60)
            print(f"Rows: True Labels, Columns: Predicted Labels")
            print(f"Class Labels: {list(multi_classifier.class_labels)}")
        
            print("\nLogistic Regression Confusion Matrix:")
            print(multi_logistic_results['confusion_matrix'])
        
            print("\nBayesian Logistic Regression Confusion Matrix:")
            print(multi_bayesian_results['confusion_matrix'])

            # --- Generate and save plots ---
            print("\nGenerating visualization plots...")
            plot_confusion_matrix_heatmap(
                multi_logistic_results['confusion_matrix'],
                multi_classifier.class_labels,
                'Logistic Regression Confusion Matrix',
                'logistic_regression_cm.png'
            )
            plot_confusion_matrix_heatmap(
                multi_bayesian_results['confusion_matrix'],
                multi_classifier.class_labels,
                'Bayesian Logistic Regression Confusion Matrix',
                'bayesian_regression_cm.png'
            )
            
            plot_roc_curves(
                multi_logistic_results['y_true'],
                multi_logistic_results['y_proba'],
                multi_classifier.class_labels,
                'Logistic Regression ROC Curves',
                'logistic_regression_roc.png'
            )
            plot_roc_curves(
                multi_bayesian_results['y_true'],
                multi_bayesian_results['y_proba'],
                multi_classifier.class_labels,
                'Bayesian Logistic Regression ROC Curves',
                'bayesian_regression_roc.png'
            )
            print("Plots saved to .png files.")

if __name__ == "__main__":
    main()