import argparse
import pandas as pd
from src.preprocess import Preprocessor
from src.binary_classifier import BinaryClassifier
from src.multiclass_classifier import MultiClassifier
from src.regression_classifier import RegressionClassifier # Add this import

def main():
    # --- 0. 인자 파싱 설정 (옵션 선택 기능) ---
    parser = argparse.ArgumentParser(description='Run Classification Model')
    parser.add_argument('--mode', type=str, choices=['binary', 'multi', 'regression'], default='binary',
                        help='Choose classification mode: "binary", "multi", or "regression"')
    parser.add_argument('--binary_b', type=float, default=1.0,
                        help='Hyperparameter b for binary Bayesian logistic regression')
    parser.add_argument('--multi_b', type=float, default=0.1,
                        help='Hyperparameter b for multiclass Bayesian logistic regression')
    parser.add_argument('--regression_b', type=float, default=10.0, 
                        help='Hyperparameter b (sigma for Normal prior) for regression Bayesian model')
    parser.add_argument('--indicator', type=bool, default=True,
                        help='Include indicator variables (>=40) in preprocessing')
    args = parser.parse_args()

    # --- 1. 데이터 로딩 및 전처리 (공통) ---
    print("--- Starting Preprocessing ---")
    preprocessor = Preprocessor(args,'mirna_v4.xlsx')
    X_combined, y_original, feature_names = preprocessor.preprocess()
    X_scaled = preprocessor.get_scaled_data()
    print("--- Preprocessing Complete ---")
    
    # --- 2. 모드에 따른 분류 모델 실행 ---
    if args.mode == 'binary':
        # === Binary Classification Logic ===
        print("\n--- Starting Binary Classification ---")
        binary_classifier = BinaryClassifier(args,X_scaled, y_original, feature_names, n_splits=5)
        
        # 일반 로지스틱 회귀 실행
        print("Running standard logistic regression...")
        logistic_results = binary_classifier.cv_logistic_regression()
        
        # 베이지안 로지스틱 회귀 실행
        print("Running Bayesian logistic regression...")
        bayesian_results = binary_classifier.cv_bayesian_logistic_regression()
        results_summary = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Score', 'AUC Score'],
            'Logistic': [logistic_results['accuracy'], logistic_results['f1_score'], logistic_results['auc_score']],
            'Bayesian': [bayesian_results['accuracy'], bayesian_results['f1_score'], bayesian_results['auc_score']]
        })
        print(results_summary.set_index('Metric').round(4))

    elif args.mode == 'multi':
        # === Multiclass Classification Logic ===
        print("\n--- Starting Multiclass Classification ---")
        multi_classifier = MultiClassifier(args,X_scaled, y_original)
        
        print("Running standard logistic regression...")
        logistic_results = multi_classifier.cv_logistic_regression()
        
        # 베이지안 로지스틱 회귀 실행
        print("Running Bayesian logistic regression...")
        bayesian_results = multi_classifier.cv_bayesian_logistic_regression()
        print("--- Multiclass Classification Complete ---")

        # 결과 요약
        print("\n" + "="*60)
        print("        Multiclass Cross-Validation Results (Average)")
        print("="*60)

        results_summary = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Score', 'AUC Score'],
            'Logistic': [logistic_results['accuracy'], logistic_results['f1_score'], logistic_results['auc_score']],
            'Bayesian': [bayesian_results['accuracy'], bayesian_results['f1_score'], bayesian_results['auc_score']]
        })
        print(results_summary.set_index('Metric').round(4))

    elif args.mode == 'regression': # regression mode
        print("\n--- Starting Regression Mode ---")
        y_regression = preprocessor.prepare_regression_target() 

        regression_classifier = RegressionClassifier(args, X_scaled, y_regression, n_splits=5)
        
        print("Running standard Ridge regression...")
        ridge_results = regression_classifier.cv_ridge_regression()
        
        print("Running Bayesian regression...")
        bayesian_regression_results = regression_classifier.cv_bayesian_regression()
        print("--- Regression Mode Complete ---")

        print("\n" + "="*60)
        print("        Regression Cross-Validation Results (Average)")
        print("="*60)

        results_summary = pd.DataFrame({
            'Metric': ['MSE', 'R2 Score', 'F1 Score', 'AUC Score'],
            'Ridge': [ridge_results['mse'], ridge_results['r2_score'], ridge_results['f1_score'], ridge_results['auc_score']],
            'Bayesian': [bayesian_regression_results['mse'], bayesian_regression_results['r2_score'], bayesian_regression_results['f1_score'], bayesian_regression_results['auc_score']]
        })
        print(results_summary.set_index('Metric').round(4))

        print("\n--- Generating AUC-like plots for Regression Mode ---")
        regression_classifier.fit_and_plot() 

if __name__ == "__main__":
    main()