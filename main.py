from src.preprocess import Preprocessor
from src.binary_classifier import BinaryClassifier
import pandas as pd

def main():
    # --- 1. 데이터 로딩 및 전처리 ---
    print("--- Starting Preprocessing ---")
    preprocessor = Preprocessor('mirna_v3.xlsx')
    X_combined, y_original = preprocessor.preprocess()
    X_scaled = preprocessor.get_scaled_data()
    print("--- Preprocessing Complete ---")
    
    # --- 2. 이진 분류 모델 실행 ---
    print("\n--- Starting Binary Classification ---")
    binary_classifier = BinaryClassifier(X_scaled, y_original)
    
    # 일반 로지스틱 회귀 실행
    print("Running standard logistic regression...")
    logistic_results = binary_classifier.run_logistic_regression()
    
    # 베이지안 로지스틱 회귀 실행
    print("Running Bayesian logistic regression...")
    bayesian_results = binary_classifier.run_bayesian_logistic_regression()
    print("--- Binary Classification Complete ---")

    # --- 3. 최종 결과 요약 ---
    print("\n" + "="*60)
    print("        Cross-Validation Results (Average)")
    print("="*60)

    results_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Recall', 'F1 Score'],
        'Logistic': [logistic_results['accuracy'], logistic_results['recall'], logistic_results['f1_score']],
        'Bayesian': [bayesian_results['accuracy'], bayesian_results['recall'], bayesian_results['f1_score']]
    })
    print(results_summary.set_index('Metric').round(4))

if __name__ == "__main__":
    main()
