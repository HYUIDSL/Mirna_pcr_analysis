import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import re

# --- 1. 데이터 로딩 및 전처리 (동일) ---
def read_xlsx(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

df = read_xlsx('mirna_v2.xlsx')
X = df.iloc[:, 19:39]
y_original = df.iloc[:, 7].copy()

# ... (X_diff_final 만드는 전처리 과정은 모두 동일) ...
filtered_columns = [col for col in X.columns if not col.startswith('dCt')]
mir_columns = [col for col in filtered_columns if 'miR' in col]
non_mir_columns = [col for col in filtered_columns if 'miR' not in col]
X_testing=X[non_mir_columns]
X_ref=X[mir_columns]


problem_columns = [col for col in X_testing.columns if (X_testing[col] == '-').any()]
valid_columns = [col for col in X_testing.columns if col not in problem_columns]
valid_indices = [X_testing.columns.get_loc(col) for col in valid_columns]
X_testing_valid = X_testing[valid_columns]
X_ref_valid = X_ref.iloc[:, valid_indices]
X_testing_numeric = X_testing_valid.astype(float)
X_ref_numeric = X_ref_valid.astype(float)


diff_values = X_testing_numeric.values - X_ref_numeric.values
new_column_names = []
for test_col, ref_col in zip(X_testing_valid.columns, X_ref_valid.columns):
    if '103' in ref_col: new_name = f"{test_col}_103"; new_name = re.sub(r'\.\d+', '', new_name)
    elif '25' in ref_col: new_name = f"{test_col}_25"; new_name = re.sub(r'\.\d+', '', new_name)
    else: new_name = f"{test_col}_ref"; new_name = re.sub(r'\.\d+', '', new_name)
    new_column_names.append(new_name)
X_diff_final = pd.DataFrame(diff_values, index=X.index, columns=new_column_names)
# 2. 40 초과 여부(0/1)가 담긴 데이터프레임
indicator_data = (X_testing_numeric.values >= 40).astype(int)
X_indicator_final = pd.DataFrame(indicator_data, index=X.index, columns=new_column_names)

# 3. 40 초과 여부 데이터프레임의 컬럼 이름 변경
#    (예: 'Feature_A' -> 'Feature_A_is_over_40')
indicator_new_cols = [f"{col}_is_geq_40" for col in X_indicator_final.columns]
X_indicator_final.columns = indicator_new_cols

print(X_indicator_final)

# 4. 두 데이터프레임을 가로(열) 방향으로 합치기
X_combined = pd.concat([X_diff_final, X_indicator_final], axis=1)

print(X_combined)

# --- 2. 이진 분류 문제 및 가중치 생성 (동일) ---
y_str = y_original.apply(lambda val: 'Normal' if str(val).startswith('C') else ('Alzheimer' if str(val).startswith('AD') else ('MCI' if str(val).startswith('MCI') else val)))
y_binary = y_str.apply(lambda val: 0 if val == 'Normal' else 1)
weights = np.ones(len(y_str))
mci_weight = 0.5
weights[y_str[y_str == 'MCI'].index] = mci_weight
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# --- 3. 모델 학습 및 평가 ---
n_features = X_scaled.shape[1]

# 3-1. 일반 로지스틱 회귀 (동일)
print(">>>>>> Processing Logistic Regression with Sample Weights <<<<<<")
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y_binary, sample_weight=weights)
y_prob_logistic = model.predict_proba(X_scaled)[:, 1]
y_pred_logistic = model.predict(X_scaled)
acc_logistic = accuracy_score(y_binary, y_pred_logistic, sample_weight=weights)
auc_logistic = roc_auc_score(y_binary, y_prob_logistic, sample_weight=weights)

# 3-2. PyMC 베이지안 회귀 (수정됨)
print("\n>>>>>> Processing Bayesian Regression with Weighted Likelihood <<<<<<")
with pm.Model() as weighted_logistic_model:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Laplace("beta", mu=0, b=0.5, shape=n_features)
    logit_p = alpha + pm.math.dot(X_scaled, beta)
    
    # <<< 핵심: 가능도(likelihood)에 가중치를 적용 >>>
    # API 변경에 따라 수정된 부분
    log_likelihood = pm.logp(pm.Bernoulli.dist(logit_p=logit_p), y_binary)
    
    weighted_log_likelihood = log_likelihood * weights
    pm.Potential('weighted_likelihood', weighted_log_likelihood.sum())
    
    trace = pm.sample(1000, tune=1000, cores=1)

# 예측 및 성능 계산 (동일)
alpha_samples = trace.posterior['alpha'].values.flatten()
beta_samples = trace.posterior['beta'].values.reshape(-1, n_features)
all_probs = [1 / (1 + np.exp(-(alpha_s + np.dot(X_scaled, beta_s)))) for alpha_s, beta_s in zip(alpha_samples, beta_samples)]
y_prob_bayesian = np.mean(all_probs, axis=0)
y_pred_bayesian = (y_prob_bayesian > 0.5).astype(int)
acc_bayesian = accuracy_score(y_binary, y_pred_bayesian, sample_weight=weights)
auc_bayesian = roc_auc_score(y_binary, y_prob_bayesian, sample_weight=weights)


# --- 4. 최종 결과 출력 (해석 추가) ---

# 4-1. 모델 성능 비교 테이블 생성
performance_df = pd.DataFrame({
    'Metric': ['AUC', 'Accuracy'],
    'Logistic': [auc_logistic, acc_logistic],
    'Bayesian': [auc_bayesian, acc_bayesian]
}).set_index('Metric')

print("\n" + "="*50)
print("     Model Performance Comparison (In-Sample, Weighted)")
print("="*50)
print(performance_df.round(4))
print("\n⚠️ 경고: 이 결과는 훈련 데이터 자체에 대한 성능으로, 과적합되었을 수 있습니다.")


# 4-2. 피처별 계수 비교 테이블 생성
summary = az.summary(trace, var_names=['beta'])
coefficients_df = pd.DataFrame({
    'Logistic_Coef': model.coef_[0],
    'Bayesian_Coef_Mean': summary['mean'].values,
    'Bayesian_94%_HDI': [f"[{l:.2f}, {h:.2f}]" for l, h in zip(summary['hdi_3%'].values, summary['hdi_97%'].values)]
}, index=X_combined.columns)

# 베이지안 계수 크기 순으로 정렬
sorted_coefficients_df = coefficients_df.reindex(
    coefficients_df['Bayesian_Coef_Mean'].abs().sort_values(ascending=False).index
)

print("\n" + "="*80)
print("                          Feature Coefficient Comparison")
print("="*80)
print(sorted_coefficients_df.round(4))


# --- 5. <<< 추가: 간단한 결과 해석 출력 >>> ---
print("\n" + "="*60)
print("                  Simple Interpretation")
print("="*60)

# 5-1. 모델 성능 해석
best_auc_model = performance_df.loc['AUC'].idxmax()
best_acc_model = performance_df.loc['Accuracy'].idxmax()

print("\n### 모델 성능 요약 (학습 데이터 기준) ###")
print(f"- AUC 점수 기준, '{best_auc_model}' 모델이 더 나은 성능을 보였습니다 (Logistic: {auc_logistic:.4f}, Bayesian: {auc_bayesian:.4f}).")
print(f"- Accuracy 점수 기준, '{best_acc_model}' 모델이 더 나은 성능을 보였습니다 (Logistic: {acc_logistic:.4f}, Bayesian: {acc_bayesian:.4f}).")

# 5-2. 주요 피처 해석
# 베이지안 모델 기준 상위 3개 피처 추출
top_3_features = sorted_coefficients_df.head(3)

print("\n### 주요 변수(Feature) 요약 (베이지안 모델 기준) ###")
for feature_name, row in top_3_features.iterrows():
    coef_mean = row['Bayesian_Coef_Mean']
    hdi = row['Bayesian_94%_HDI']
    
    # 신뢰구간이 0을 포함하는지 확인
    hdi_lower, hdi_upper = map(float, hdi.strip('[]').split(','))
    
    if hdi_lower > 0:
        direction = "양(+)"
        significance = "통계적으로 유의미한"
    elif hdi_upper < 0:
        direction = "음(-)"
        significance = "통계적으로 유의미한"
    else:
        direction = ""
        significance = "통계적으로 유의미하지 않은"
    
    print(f"- '{feature_name}': {significance} {direction}의 연관성을 보입니다 (계수 평균: {coef_mean:.4f}, 94% 신뢰구간: {hdi}).")
sample_index = 0
sample_to_predict = X_scaled[sample_index:sample_index+1] # 모델 입력을 위해 2D 배열 형태 유지
true_label_code = y_binary[sample_index]
true_label_str = y_str[sample_index]

print("\n" + "="*50)
print(f"      Prediction for Sample #{sample_index}")
print("="*50)
print(f"True Label: {true_label_str} (Code: {true_label_code})")

# 4-1. 일반 로지스틱 회귀 예측
prob_logistic = model.predict_proba(sample_to_predict)[:, 1][0]
print("\n### Logistic Regression Prediction ###")
print(f"비정상(MCI/AD)일 확률: {prob_logistic:.2%}")
print(f"  -> 최종 예측: {'비정상' if prob_logistic > 0.5 else '정상'}")

# 4-2. 베이지안 회귀 예측
alpha_samples = trace.posterior['alpha'].values.flatten()
beta_samples = trace.posterior['beta'].values.reshape(-1, n_features)
all_probs = [1 / (1 + np.exp(-(alpha_s + np.dot(sample_to_predict, beta_s)))) for alpha_s, beta_s in zip(alpha_samples, beta_samples)]
all_probs = np.array(all_probs).flatten()

y_prob_bayesian_mean = all_probs.mean()
hdi = az.hdi(all_probs, hdi_prob=0.94)

print("\n### Bayesian Regression Prediction ###")
print(f"비정상(MCI/AD)일 확률 (평균): {y_prob_bayesian_mean:.2%}")
print(f"  -> 94% 신뢰구간: [{hdi[0]:.2%}, {hdi[1]:.2%}]")
print(f"  -> 최종 예측: {'비정상' if y_prob_bayesian_mean > 0.5 else '정상'}")
print("="*50)