import argparse
import pandas as pd
from src.preprocess import Preprocessor
from src.binary_classifier import BinaryClassifier, plot_binary_comparison
from src.multiclass_classifier import MultiClassifier, plot_multiclass_comparison
from src.regression_classifier import RegressionClassifier, plot_regression_comparison

def main():
    # --- 0. 인자 파싱 설정 (옵션 선택 기능) ---
    parser = argparse.ArgumentParser(description='Run Classification Model')
    parser.add_argument('--mode', type=str, choices=['binary', 'multi', 'regression', 'all'], default='all',
                        help='Choose classification mode: "binary", "multi", "regression", or "all" to run all modes.')
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
    preprocessor = Preprocessor(args, 'mirna_v4.xlsx')
    X_combined, y_original, feature_names = preprocessor.preprocess()
    X_scaled = preprocessor.get_scaled_data()
    print("--- Preprocessing Complete ---")

    # --- 2. 실행할 모드 결정 ---
    if args.mode == 'all':
        modes_to_run = ['binary', 'multi', 'regression']
    else:
        modes_to_run = [args.mode]

    # --- 3. 선택된 모드들에 대해 분류 모델 실행 ---
    for mode in modes_to_run:
        if mode == 'binary':
            print("\n" + "="*80)
            print(" " * 25 + "--- Starting Binary Classification ---")
            print("="*80)
            binary_classifier = BinaryClassifier(args, X_scaled, y_original, feature_names, n_splits=5)
            
            print("\nRunning Cross-Validation for Logistic and Bayesian models...")
            logistic_metrics, y_true_cv, y_pred_logistic_cv = binary_classifier.cv_logistic_regression()
            bayesian_metrics, _, y_pred_bayesian_cv = binary_classifier.cv_bayesian_logistic_regression()

            print("\n--- Generating ROC curve for CV results ---")
            plot_binary_comparison(y_true_cv, {'Logistic': y_pred_logistic_cv, 'Bayesian': y_pred_bayesian_cv}, suffix="_CV")
            
            print("\n" + "-"*40)
            print("     Binary CV Results (Average)")
            print("-"*40)
            results_summary = pd.DataFrame({
                'Metric': ['Accuracy', 'F1 Score', 'AUC Score'],
                'Logistic': [logistic_metrics['accuracy'], logistic_metrics['f1_score'], logistic_metrics['auc_score']],
                'Bayesian': [bayesian_metrics['accuracy'], bayesian_metrics['f1_score'], bayesian_metrics['auc_score']]
            })
            print(results_summary.set_index('Metric').round(4))
            print("-" * 40)

            print("\n--- Fitting final models and generating ROC curve for Full Dataset ---")
            y_true_full, logistic_preds_full, bayesian_preds_full = binary_classifier.fit_and_predict_full()
            plot_binary_comparison(y_true_full, {'Logistic': logistic_preds_full, 'Bayesian': bayesian_preds_full}, suffix="_Full_Dataset")

        elif mode == 'multi':
            print("\n" + "="*80)
            print(" " * 23 + "--- Starting Multiclass Classification ---")
            print("="*80)
            multi_classifier = MultiClassifier(args, X_scaled, y_original)
            
            print("\nRunning Cross-Validation for Logistic and Bayesian models...")
            logistic_metrics, y_true_cv, y_pred_logistic_cv = multi_classifier.cv_logistic_regression()
            bayesian_metrics, _, y_pred_bayesian_cv = multi_classifier.cv_bayesian_logistic_regression()

            print("\n--- Generating ROC curves for CV results ---")
            plot_multiclass_comparison(y_true_cv, {'Logistic': y_pred_logistic_cv, 'Bayesian': y_pred_bayesian_cv}, multi_classifier.class_labels, suffix="_CV")

            print("\n" + "-"*40)
            print("   Multiclass CV Results (Average)")
            print("-"*40)
            results_summary = pd.DataFrame({
                'Metric': ['Accuracy', 'F1 Score', 'AUC Score'],
                'Logistic': [logistic_metrics['accuracy'], logistic_metrics['f1_score'], logistic_metrics['auc_score']],
                'Bayesian': [bayesian_metrics['accuracy'], bayesian_metrics['f1_score'], bayesian_metrics['auc_score']]
            })
            print(results_summary.set_index('Metric').round(4))
            print("-" * 40)

            print("\n--- Fitting final models and generating ROC curves for Full Dataset ---")
            y_true_full, logistic_preds_full, bayesian_preds_full = multi_classifier.fit_and_predict_full()
            plot_multiclass_comparison(y_true_full, {'Logistic': logistic_preds_full, 'Bayesian': bayesian_preds_full}, multi_classifier.class_labels, suffix="_Full_Dataset")

        elif mode == 'regression':
            print("\n" + "="*80)
            print(" " * 28 + "--- Starting Regression Mode ---")
            print("="*80)
            y_regression = preprocessor.prepare_regression_target() 

            regression_classifier = RegressionClassifier(args, X_scaled, y_regression, n_splits=5)
            
            print("\nRunning Cross-Validation for Ridge and Bayesian regression...")
            ridge_metrics, y_true_cv, y_pred_ridge_cv = regression_classifier.cv_ridge_regression()
            bayesian_metrics, _, y_pred_bayesian_cv = regression_classifier.cv_bayesian_regression()
            
            print("\n--- Generating ROC curves for CV results ---")
            plot_regression_comparison(y_pred_ridge_cv, y_pred_bayesian_cv, y_true_cv, suffix="_CV")

            print("\n" + "-"*50)
            print("        Regression CV Results (Average)")
            print("-"*50)
            results_summary = pd.DataFrame({
                'Metric': [
                    'MSE', 
                    'R2 Score', 
                    'F1 (AD vs Rest)', 
                    'AUC (AD vs Rest)',
                    'F1 (Normal vs Rest)',
                    'AUC (Normal vs Rest)'
                ],
                'Ridge': [
                    ridge_metrics['mse'], 
                    ridge_metrics['r2_score'], 
                    ridge_metrics['f1_score_ad'], 
                    ridge_metrics['auc_score_ad'],
                    ridge_metrics['f1_score_normal'],
                    ridge_metrics['auc_score_normal']
                ],
                'Bayesian': [
                    bayesian_metrics['mse'], 
                    bayesian_metrics['r2_score'], 
                    bayesian_metrics['f1_score_ad'], 
                    bayesian_metrics['auc_score_ad'],
                    bayesian_metrics['f1_score_normal'],
                    bayesian_metrics['auc_score_normal']
                ]
            })
            print(results_summary.set_index('Metric').round(4))
            print("-"*50)

            print("\n--- Fitting final models and generating ROC curves for Full Dataset ---")
            y_true_full, y_pred_ridge_full, y_pred_bayesian_full = regression_classifier.fit_and_predict_full()
            plot_regression_comparison(y_pred_ridge_full, y_pred_bayesian_full, y_true_full, suffix="_Full_Dataset")

if __name__ == "__main__":
    main()