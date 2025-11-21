import argparse
import pandas as pd
from src.preprocess import Preprocessor
from src.binary_classifier import BinaryClassifier
from src.multiclass_classifier import MultiClassifier

def main():
    # --- 0. 인자 파싱 설정 (옵션 선택 기능) ---
    parser = argparse.ArgumentParser(description='Run Classification Model')
    parser.add_argument('--mode', type=str, choices=['binary', 'multi'], default='binary',
                        help='Choose classification mode: "binary" or "multi"')
    parser.add_argument('--binary_b', type=float, default=0.5,
                        help='Hyperparameter b for binary Bayesian logistic regression')
    parser.add_argument('--multi_b', type=float, default=0.1,
                        help='Hyperparameter b for multiclass Bayesian logistic regression')
    args = parser.parse_args()

    # --- 1. 데이터 로딩 및 전처리 (공통) ---
    print("--- Starting Preprocessing ---")
    preprocessor = Preprocessor('mirna_v3.xlsx')
    X_combined, y_original = preprocessor.preprocess()
    X_scaled = preprocessor.get_scaled_data()
    print("--- Preprocessing Complete ---")
    
    # --- 2. 모드에 따른 분류 모델 실행 ---
    if args.mode == 'binary':
        # === Binary Classification Logic ===
        print("\n--- Starting Binary Classification ---")
        binary_classifier = BinaryClassifier(args,X_scaled, y_original)
        
        # 일반 로지스틱 회귀 실행
        print("Running standard logistic regression...")
        logistic_results = binary_classifier.cv_logistic_regression()
        
        # 베이지안 로지스틱 회귀 실행
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
        multi_classifier = MultiClassifier(args,X_scaled, y_original)
        
        # [주의] MultiClassifier 클래스 내부의 실제 실행 메서드 이름으로 변경해야 합니다.
        # 예: cv_multiclass_logistic(), run_model() 등 작성하신 메서드를 호출하세요.
        print("Running Multiclass Classification...")
        
        # 예시 코드 (실제 메서드명에 맞춰 수정 필요)
        multi_results = multi_classifier.run_classification() 
        
        print("--- Multiclass Classification Complete ---")

        # 결과 요약 (Multiclass 결과 형식에 맞춰 수정 필요)
        print("\n" + "="*60)
        print("        Multiclass Cross-Validation Results")
        print("="*60)
        print(multi_results)

if __name__ == "__main__":
    main()