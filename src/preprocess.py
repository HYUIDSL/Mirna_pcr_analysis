import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self,args, file_path):
        self.args = args
        self.file_path = file_path
        self.df = None
        self.X_combined = None
        self.y_original = None

    def load_data(self):
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=0)

            return self.df
        except Exception as e:
            print(f"Error reading {self.file_path}: {e}")
            return None

    def preprocess(self):

        self.df= self.load_data()

        X_raw = self.df.iloc[:, 19:37]
        self.y_original = self.df.iloc[:, 8].copy()

        filtered_columns = [col for col in X_raw.columns if not col.startswith('dCt')]
        mir_columns = [col for col in filtered_columns if 'miR' in col]
        non_mir_columns = [col for col in filtered_columns if 'miR' not in col]

        if not all(col in X_raw.columns for col in non_mir_columns):
            raise ValueError("non_mir_columns contains columns not found in X_raw")
        if not all(col in X_raw.columns for col in mir_columns):
            raise ValueError("mir_columns contains columns not found in X_raw")

        X_testing_raw = X_raw[non_mir_columns]
        X_ref_raw = X_raw[mir_columns]

        problem_columns = [col for col in X_testing_raw.columns if (X_testing_raw[col] == '-').any()]
        valid_columns = [col for col in X_testing_raw.columns if col not in problem_columns]

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
            if '103' in ref_col:
                new_name = f"{test_col}_103"
                new_name = re.sub(r'\.\d+', '', new_name)
            elif '25' in ref_col:
                new_name = f"{test_col}_25"
                new_name = re.sub(r'\.\d+', '', new_name)
            else:
                new_name = f"{test_col}_ref"
                new_name = re.sub(r'\.\d+', '', new_name)
            new_column_names.append(new_name)

        X_diff_final = pd.DataFrame(diff_values, index=self.df.index[:len(diff_values)], columns=new_column_names)

        indicator_data = (X_testing_numeric.values >= 40).astype(int)
        X_indicator_final = pd.DataFrame(indicator_data, index=X_diff_final.index, columns=new_column_names)
        indicator_new_cols = [f"{col}_is_geq_40" for col in X_indicator_final.columns]
        X_indicator_final.columns = indicator_new_cols
        self.X_combined = pd.concat([X_diff_final, X_indicator_final], axis=1)

        if self.args.indicator:
            indicator_data = (X_testing_numeric.values >= 40).astype(int)
            X_indicator_final = pd.DataFrame(indicator_data, index=X_diff_final.index, columns=new_column_names)
            indicator_new_cols = [f"{col}_is_geq_40" for col in X_indicator_final.columns]
            X_indicator_final.columns = indicator_new_cols
            self.X_combined = pd.concat([X_diff_final, X_indicator_final], axis=1)
        else:
            self.X_combined = X_diff_final
        
        return self.X_combined, self.y_original, self.X_combined.columns

    def get_scaled_data(self):
        if self.X_combined is None:
            self.preprocess()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_combined)
        return X_scaled

    def prepare_regression_target(self):
        if self.y_original is None:
            self.preprocess()
        
        y_str = self.y_original.apply(lambda val: 'Normal' if str(val).startswith('C') else ('Alzheimer' if str(val).startswith('AD') else 'MCI'))
        
        y_regression = pd.Series(index=y_str.index, dtype=float)
        
        y_regression[y_str == 'Normal'] = 0.0
        y_regression[y_str == 'Alzheimer'] = 1.0
        
        mci_indices = y_str[y_str == 'MCI'].index
        num_mci = len(mci_indices)
        y_regression.loc[mci_indices] = 0.35
        
        return y_regression
