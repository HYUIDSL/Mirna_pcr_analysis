import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineer:
    """
    A class to perform feature engineering on the preprocessed miRNA data.
    """
    def __init__(self, data):
        """
        Initializes the FeatureEngineer.

        Args:
            data (pd.DataFrame): The preprocessed data from the Preprocessor class.
        """
        self.data = data.copy()

    def create_polynomial_features(self):
        """
        Creates polynomial and interaction features from the ddCt columns.
        """
        print("--- Starting Feature Engineering: Polynomial & Interaction Features ---")
        
        # Separate the ddCt columns and the indicator columns
        ddct_cols = [col for col in self.data.columns if not col.endswith('_is_geq_40')]
        indicator_cols = [col for col in self.data.columns if col.endswith('_is_geq_40')]
        
        ddct_data = self.data[ddct_cols]
        indicator_data = self.data[indicator_cols]
        
        # Create polynomial and interaction features (degree=2)
        # include_bias=False avoids adding a column of ones.
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        
        poly_features = poly.fit_transform(ddct_data)
        
        # Create new column names for the polynomial features
        poly_feature_names = poly.get_feature_names_out(ddct_cols)
        
        # Create a new DataFrame with the polynomial features
        poly_df = pd.DataFrame(poly_features, index=self.data.index, columns=poly_feature_names)
        
        print(f"Created {poly_df.shape[1] - len(ddct_cols)} new polynomial/interaction features.")
        
        # Combine the new polynomial features with the original indicator features
        self.data = pd.concat([poly_df, indicator_data], axis=1)
        
        return self.data

    def transform(self):
        """
        Applies all feature engineering steps.
        """
        self.data = self.create_polynomial_features()
        # Future engineering methods can be added here
        
        print("--- Feature Engineering Complete ---")
        return self.data
