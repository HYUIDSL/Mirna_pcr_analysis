import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score # <<< F1 Score 추가
from sklearn.model_selection import KFold
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 데이터 로딩 및 전처리 (동일) ---
def read_xlsx(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# 파일 이름 확인 및 실제 경로로 수정 필요
df = read_xlsx('mirna_v2.xlsx')
# 실제 데이터 구조에 맞게 인덱스 조정 필요 (예시 값)
X_raw = df.iloc[:, 19:39]
y_original = df.iloc[:, 7].copy()

# --- X_diff_final 및 X_indicator_final 생성 ---
filtered_columns = [col for col in X_raw.columns if not col.startswith('dCt')]
mir_columns = [col for col in filtered_columns if 'miR' in col]
non_mir_columns = [col for col in filtered_columns if 'miR' not in col]
# 열 이름 존재 확인 및 에러 처리 추가
if not all(col in X_raw.columns for col in non_mir_columns):
    raise ValueError("non_mir_columns contains columns not found in X_raw")
if not all(col in X_raw.columns for col in mir_columns):
     raise ValueError("mir_columns contains columns not found in X_raw")

X_testing_raw = X_raw[non_mir_columns]
X_ref_raw = X_raw[mir_columns]

problem_columns = [col for col in X_testing_raw.columns if (X_testing_raw[col] == '-').any()]
valid_columns = [col for col in X_testing_raw.columns if col not in problem_columns]

# valid_columns가 비어있는 경우 에러 처리
if not valid_columns:
    raise ValueError("No valid columns found after filtering for '-' values.")

valid_indices = [X_testing_raw.columns.get_loc(col) for col in valid_columns]
X_testing_valid = X_testing_raw[valid_columns]
X_ref_valid = X_ref_raw.iloc[:, valid_indices]

X_testing_numeric = X_testing_valid.astype(float)
X_ref_numeric = X_ref_valid.astype(float)
diff_values = X_testing_numeric.values - X_ref_numeric.values

new_column_names = []
for test_col, ref_col in zip(X_testing_valid.columns, X_ref_valid.columns):
    if '103' in ref_col: new_name = f"{test_col}_103"; new_name = re.sub(r'\.\d+', '', new_name)
    elif '25' in ref_col: new_name = f"{test_col}_25"; new_name = re.sub(r'\.\d+', '', new_name)
    else: new_name = f"{test_col}_ref"; new_name = re.sub(r'\.\d+', '', new_name)
    new_column_names.append(new_name)

# Handle potential length mismatch if df index has gaps
X_diff_final = pd.DataFrame(diff_values, index=df.index[:len(diff_values)], columns=new_column_names) # Index 정렬

# --- 지시 변수 추가 ---
indicator_data = (X_testing_numeric.values >= 40).astype(int)
X_indicator_final = pd.DataFrame(indicator_data, index=X_diff_final.index, columns=new_column_names)
indicator_new_cols = [f"{col}_is_geq_40" for col in X_indicator_final.columns]
X_indicator_final.columns = indicator_new_cols
X_combined = pd.concat([X_diff_final, X_indicator_final], axis=1)

# --- 2. 이진 분류 문제, 가중치 생성 및 데이터 스케일링 (동일) ---
y_str = y_original.apply(lambda val: 'Normal' if str(val).startswith('C') else ('Alzheimer' if str(val).startswith('AD') else ('MCI' if str(val).startswith('MCI') else val)))
y_binary = y_str.apply(lambda val: 0 if val == 'Normal' else 1).values # Numpy 배열로 변환
weights = np.ones(len(y_str))
mci_weight = 0.8
weights[y_str[y_str == 'MCI'].index] = mci_weight

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# --- 3. 5-Fold Cross-Validation 설정 ---
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 각 폴드의 성능 지표를 저장할 리스트
logistic_acc_scores = []
logistic_recall_scores = []
logistic_f1_scores = [] # <<< F1 Score 리스트 추가
bayesian_acc_scores = []
bayesian_recall_scores = []
bayesian_f1_scores = [] # <<< F1 Score 리스트 추가

n_features = X_scaled.shape[1]
fold_count = 1

# --- CV 루프 시작 ---
for train_index, val_index in kf.split(X_scaled):
    print(f"\n>>>>>> Processing Fold {fold_count}/{n_splits} <<<<<<")

    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y_binary[train_index], y_binary[val_index]
    weights_train, weights_val = weights[train_index], weights[val_index]

    # --- 3-1. 일반 로지스틱 회귀 ---
    model_logistic = LogisticRegression(max_iter=1000)
    model_logistic.fit(X_train, y_train, sample_weight=weights_train)
    y_pred_logistic = model_logistic.predict(X_val)

    # 성능 계산
    acc_logistic = accuracy_score(y_val, y_pred_logistic, sample_weight=weights_val)
    recall_logistic = recall_score(y_val, y_pred_logistic, sample_weight=weights_val, zero_division=0)
    f1_logistic = f1_score(y_val, y_pred_logistic, sample_weight=weights_val, zero_division=0) # <<< F1 계산
    logistic_acc_scores.append(acc_logistic)
    logistic_recall_scores.append(recall_logistic)
    logistic_f1_scores.append(f1_logistic) # <<< 저장

    # --- 3-2. PyMC 베이지안 회귀 ---
    print(f"  Training Bayesian model for Fold {fold_count}...")
    with pm.Model() as weighted_logistic_model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Laplace("beta", mu=0, b=0.5, shape=n_features)
        logit_p = alpha + pm.math.dot(X_train, beta)
        log_likelihood = pm.logp(pm.Bernoulli.dist(logit_p=logit_p), y_train)
        weighted_log_likelihood = log_likelihood * weights_train
        pm.Potential('weighted_likelihood', weighted_log_likelihood.sum())
        trace = pm.sample(500, tune=500, cores=1, progressbar=False, return_inferencedata=False)

    # 예측
    alpha_samples = trace['alpha']
    beta_samples = trace['beta']
    all_probs = [1 / (1 + np.exp(-(alpha_s + np.dot(X_val, beta_s)))) for alpha_s, beta_s in zip(alpha_samples, beta_samples)]
    y_prob_bayesian = np.mean(all_probs, axis=0)
    y_pred_bayesian = (y_prob_bayesian > 0.5).astype(int)

    # 성능 계산
    acc_bayesian = accuracy_score(y_val, y_pred_bayesian, sample_weight=weights_val)
    recall_bayesian = recall_score(y_val, y_pred_bayesian, sample_weight=weights_val, zero_division=0)
    f1_bayesian = f1_score(y_val, y_pred_bayesian, sample_weight=weights_val, zero_division=0) # <<< F1 계산
    bayesian_acc_scores.append(acc_bayesian)
    bayesian_recall_scores.append(recall_bayesian)
    bayesian_f1_scores.append(f1_bayesian) # <<< 저장

    fold_count += 1
# --- CV 루프 종료 ---

# --- 4. 최종 결과 요약 ---
print("\n" + "="*60)
print("        5-Fold Cross-Validation Results (Average)")
print("="*60)

results_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'F1 Score'], # <<< F1 Score 추가
    'Logistic': [np.mean(logistic_acc_scores), np.mean(logistic_recall_scores), np.mean(logistic_f1_scores)], # <<< F1 평균 추가
    'Bayesian': [np.mean(bayesian_acc_scores), np.mean(bayesian_recall_scores), np.mean(bayesian_f1_scores)] # <<< F1 평균 추가
})
print(results_summary.set_index('Metric').round(4))

print("\n--- Individual Fold Scores ---")
print("Logistic Accuracy:", [round(s, 4) for s in logistic_acc_scores])
print("Bayesian Accuracy:", [round(s, 4) for s in bayesian_acc_scores])
print("Logistic Recall:", [round(s, 4) for s in logistic_recall_scores])
print("Bayesian Recall:", [round(s, 4) for s in bayesian_recall_scores])
print("Logistic F1 Score:", [round(s, 4) for s in logistic_f1_scores]) # <<< F1 개별 점수 추가
print("Bayesian F1 Score:", [round(s, 4) for s in bayesian_f1_scores]) # <<< F1 개별 점수 추가