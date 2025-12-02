import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import KFold
import pymc as pm
import arviz as az  # HDI 계산을 위한 핵심 라이브러리
from scipy import stats
import pandas as pd
import statsmodels.api as sm

class BinaryClassifier:
    def __init__(self,args, X, y, feature_names, n_splits=10, random_state=42):
        self.args = args
        self.X = X
        self.y_original = y
        self.feature_names = feature_names
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.y_binary, self.weights = self._prepare_target()

    def _prepare_target(self):
        y_str = self.y_original.apply(lambda val: 'Normal' if str(val).startswith('C') else ('Alzheimer' if str(val).startswith('AD') else ('MCI')))
        y_binary = y_str.apply(lambda val: 1 if val == 'Alzheimer' else 0)
        weights = np.ones(len(y_str))
        mci_weight = 0.8
        weights[y_str[y_str == 'MCI'].index] = mci_weight
        return y_binary, weights

    def cv_logistic_regression(self):
        acc_scores, recall_scores, f1_scores = [], [], []

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y_binary.iloc[train_index], self.y_binary.iloc[val_index]
            weights_train, weights_val = self.weights[train_index], self.weights[val_index]

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = model.predict(X_val)

            acc_scores.append(accuracy_score(y_val, y_pred, sample_weight=weights_val))
            recall_scores.append(recall_score(y_val, y_pred, sample_weight=weights_val, zero_division=0))
            f1_scores.append(f1_score(y_val, y_pred, sample_weight=weights_val, zero_division=0))

        return {'accuracy': np.mean(acc_scores), 'recall': np.mean(recall_scores), 'f1_score': np.mean(f1_scores)}

    def cv_bayesian_logistic_regression(self):
        acc_scores, recall_scores, f1_scores = [], [], []
        n_features = self.X.shape[1]

        for train_index, val_index in self.kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y_binary.iloc[train_index], self.y_binary.iloc[val_index]
            weights_train, weights_val = self.weights[train_index], self.weights[val_index]

            with pm.Model() as weighted_logistic_model:
                alpha = pm.Normal("alpha", mu=0, sigma=1)
                beta = pm.Laplace("beta", mu=0, b=self.args.binary_b, shape=n_features)
                logit_p = alpha + pm.math.dot(X_train, beta)
                log_likelihood = pm.logp(pm.Bernoulli.dist(logit_p=logit_p), y_train)
                weighted_log_likelihood = log_likelihood * weights_train
                pm.Potential('weighted_likelihood', weighted_log_likelihood.sum())
                trace = pm.sample(500, tune=500, cores=1, progressbar=False, return_inferencedata=False)

            alpha_samples = trace['alpha']
            beta_samples = trace['beta']
            all_probs = [1 / (1 + np.exp(-(alpha_s + np.dot(X_val, beta_s)))) for alpha_s, beta_s in zip(alpha_samples, beta_samples)]
            y_prob_bayesian = np.mean(all_probs, axis=0)
            y_pred_bayesian = (y_prob_bayesian > 0.5).astype(int)

            acc_scores.append(accuracy_score(y_val, y_pred_bayesian, sample_weight=weights_val))
            recall_scores.append(recall_score(y_val, y_pred_bayesian, sample_weight=weights_val, zero_division=0))
            f1_scores.append(f1_score(y_val, y_pred_bayesian, sample_weight=weights_val, zero_division=0))

        return {'accuracy': np.mean(acc_scores), 'recall': np.mean(recall_scores), 'f1_score': np.mean(f1_scores)}

    def fit_final_models(self, draws=500, tune=500):
        """
        Fits both standard and Bayesian logistic regression models on the entire dataset.
        """
        # Standard Logistic Regression
        self.final_logistic_model = LogisticRegression(max_iter=1000)
        self.final_logistic_model.fit(self.X, self.y_binary, sample_weight=self.weights)

        # Bayesian Logistic Regression
        n_features = self.X.shape[1]
        
        with pm.Model() as self.final_bayesian_model:
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            beta = pm.Laplace("beta", mu=0, b=self.args.binary_b, shape=n_features)
            logit_p = alpha + pm.math.dot(self.X, beta)
            log_likelihood = pm.logp(pm.Bernoulli.dist(logit_p=logit_p), self.y_binary)
            weighted_log_likelihood = log_likelihood * self.weights
            pm.Potential('weighted_likelihood', weighted_log_likelihood.sum())
            
            self.final_trace = pm.sample(draws, tune=tune, cores=1, progressbar=False, return_inferencedata=False)
    def calculate_p_values(self):
        """
        Calculates p-values using Statsmodels (Frequentist) and Posterior Analysis (Bayesian).
        """
        if not hasattr(self, 'final_trace'):
            raise ValueError("Bayesian model has not been fitted. Call fit_final_models() first.")

        # --- 1. Frequentist P-values (using Statsmodels) ---
        # Statsmodels는 상수항(Intercept)을 자동으로 추가하지 않으므로 수동으로 추가해야 함
        X_design = sm.add_constant(self.X)
        
        # 가중치 적용: GLM family=Binomial 사용
        # 주의: statsmodels의 freq_weights는 정수를 기대하지만, 
        # class weight 용도로 var_weights 등을 응용하거나 
        # 단순히 가중치 없이 경향성만 보려면 weights 인자를 생략하기도 함.
        # 여기서는 가장 유사한 동작을 위해 GLM을 사용합니다.
        try:
            # statsmodels의 Logit 혹은 GLM 사용
            glm_model = sm.GLM(self.y_binary, X_design, 
                            family=sm.families.Binomial(),
                            freq_weights=self.weights) # freq_weights 사용 시 주의 (보통 빈도수)
            
            # 만약 weights가 단순 class balancing 용도(0.8 등)라면 
            # statsmodels에서는 수렴이 어려울 수 있어, weights 없이 돌리는 경우도 많습니다.
            # 여기서는 사용자 의도에 맞춰 weights를 넣습니다.
            result = glm_model.fit()
            
            freq_coefs = result.params.values
            freq_p_values = result.pvalues.values
            
        except Exception as e:
            print(f"Statsmodels fitting failed: {e}")
            # 실패 시 NaN 처리
            freq_coefs = [np.nan] * X_design.shape[1]
            freq_p_values = [np.nan] * X_design.shape[1]

        alpha_samples = self.final_trace['alpha']
        beta_samples = self.final_trace['beta']
        
        # (n_samples, n_features + 1) 형태로 결합
        all_bayes_samples = np.hstack([alpha_samples.reshape(-1, 1), beta_samples])
        hdi_prob=0.95
        
        bayes_coef_means = []
        hdi_lower = []
        hdi_upper = []
        prob_direction = [] # (선택 사항) PD: Probability of Direction

        for i in range(all_bayes_samples.shape[1]):
            param_samples = all_bayes_samples[:, i]
            
            # 1. 평균 (Posterior Mean)
            bayes_coef_means.append(np.mean(param_samples))
            
            # 2. HDI 계산 (ArviZ 사용)
            # hdi 함수는 [lower, upper] 형태의 배열을 반환
            hdi_val = az.hdi(param_samples, hdi_prob=hdi_prob)
            hdi_lower.append(hdi_val[0])
            hdi_upper.append(hdi_val[1])
            
            # 3. (참고용) P-value 대체재: pd (Probability of Direction)
            # 계수의 부호가 한쪽으로 쏠려있는 비율 (95% 이상이면 유의하다고 봄)
            prob_gt_0 = np.mean(param_samples > 0)
            prob_lt_0 = np.mean(param_samples < 0)
            prob_direction.append(max(prob_gt_0, prob_lt_0))

        # =========================================================
        # [Part 3] 결과 정리
        # =========================================================
        
        if hasattr(self.X, 'columns'):
             feature_names = ['Intercept'] + list(self.X.columns)
        else:
             feature_names = ['Intercept'] + [i for i in self.feature_names]

        results = pd.DataFrame({
            'Feature': feature_names,
            # Frequentist
            'Freq_Coef': freq_coefs,
            'Freq_P_Value': freq_p_values,
            # Bayesian
            'Bayes_Coef_Mean': bayes_coef_means,
            f'Bayes_HDI_{int(hdi_prob*100)}%_Lower': hdi_lower,
            f'Bayes_HDI_{int(hdi_prob*100)}%_Upper': hdi_upper,
            'Prob_Direction': prob_direction
        })
        
        # HDI 구간 안에 0이 포함되는지 여부 (False면 유의미함)
        lower_col = f'Bayes_HDI_{int(hdi_prob*100)}%_Lower'
        upper_col = f'Bayes_HDI_{int(hdi_prob*100)}%_Upper'
        
        # 0이 구간 밖에 있으면 True (Significant), 안에 있으면 False
        results['Bayes_Significant'] = results.apply(
            lambda x: not (x[lower_col] <= 0 <= x[upper_col]), axis=1
        )
        
        return results.round(4)

    def predict_proba(self, sample_data):
        """
        Predicts probabilities for sample data using both models.
        
        Args:
            sample_data: numpy array or pandas DataFrame of shape (n_samples, n_features)
            
        Returns:
            dict containing 'logistic_proba' and 'bayesian_proba' (probability of positive class)
        """
        if hasattr(sample_data, 'values'):
            sample_data = sample_data.values
            
        if not hasattr(self, 'final_logistic_model') or not hasattr(self, 'final_trace'):
            raise ValueError("Models have not been fitted. Call fit_final_models() first.")

        # Standard Logistic Regression Prediction
        # predict_proba returns [prob_0, prob_1], we want prob_1
        logistic_proba = self.final_logistic_model.predict_proba(sample_data)[:, 1]

        # Bayesian Logistic Regression Prediction
        alpha_samples = self.final_trace['alpha']
        beta_samples = self.final_trace['beta']
        
        probs_list = []
        for i in range(len(alpha_samples)):
            a = alpha_samples[i]
            b = beta_samples[i]
            logit = a + np.dot(sample_data, b)
            p_val = 1 / (1 + np.exp(-logit))
            probs_list.append(p_val)
        
        bayesian_proba = np.mean(probs_list, axis=0)
        
        return {
            'logistic_proba': logistic_proba,
            'bayesian_proba': bayesian_proba
        }
