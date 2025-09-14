import pandas as pd
import numpy as np
import pickle

class Preprocessor:
    def __init__(self, categorical_columns_to_encode=None):
        # Imputation statistics
        self.imputation_stats = {}
        
        # Column information
        self.feature_columns = None
        self.missing_columns = None
        
        # Encoding information
        self.categorical_columns_to_encode = categorical_columns_to_encode or []
        
        # Imputation summary from training
        self.imputation_summary = {}
        
    def extended_imputation(self, df, is_training=True):
        #Apply extended imputation with tracking columns
        # Create a copy
        df_imputed = df.copy()
        
        # Get columns with missing values
        missing_columns = df.columns[df.isnull().any()].tolist()
        
        if is_training:
            self.missing_columns = missing_columns
            self.imputation_summary = {}
        
        for col in missing_columns:
            # Count original missing values
            original_missing = df[col].isnull().sum()
            missing_percentage = (original_missing / len(df)) * 100
            
            # Create tracking column (1 where values were missing, 0 otherwise)
            tracking_col_name = f"{col}_was_imputed"
            df_imputed[tracking_col_name] = df[col].isnull().astype(int)
            
            # Determine imputation strategy based on data type
            if df[col].dtype == 'object':
                # Categorical data - use mode
                if is_training:
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        fill_value = mode_value[0]
                        method = 'mode'
                    else:
                        fill_value = 'N/A'
                        method = 'default_string'
                    
                    # Store for future use
                    self.imputation_stats[col] = {
                        'method': method,
                        'fill_value': fill_value
                    }
                else:
                    # Use stored values for transform
                    if col in self.imputation_stats:
                        fill_value = self.imputation_stats[col]['fill_value']
                        method = self.imputation_stats[col]['method']
                    else:
                        # Fallback if column wasn't missing in training
                        fill_value = 'N/A'
                        method = 'default_string'
            
            else:
                # Numerical data - use median
                if is_training:
                    fill_value = df[col].median()
                    method = 'median'
                    
                    # Store for future use
                    self.imputation_stats[col] = {
                        'method': method,
                        'fill_value': fill_value
                    }
                else:
                    # Use stored values for transform
                    if col in self.imputation_stats:
                        fill_value = self.imputation_stats[col]['fill_value']
                        method = self.imputation_stats[col]['method']
                    else:
                        # Fallback if column wasn't missing in training
                        fill_value = df[col].median()
                        method = 'median'
            
            # Apply imputation
            df_imputed[col] = df_imputed[col].fillna(fill_value)
            
            # Verify imputation worked
            remaining_missing = df_imputed[col].isnull().sum()
            imputed_count = original_missing - remaining_missing
            
            # Store summary (for training or tracking)
            if is_training:
                self.imputation_summary[col] = {
                    'original_missing': original_missing,
                    'missing_percentage': missing_percentage,
                    'method': method,
                    'fill_value': fill_value,
                    'imputed_count': imputed_count,
                    'tracking_column': tracking_col_name
                }
        
        return df_imputed
    
    def fit_transform(self, df, target_col):
        
        #Fit the preprocessor and transform the training data
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Handle missing values in target
        df = df.dropna(subset=[target_col])
        
        # Convert target to numeric, coercing errors to NaN
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        
        # Drop rows with NaN in target
        df = df.dropna(subset=[target_col])
        
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Apply extended imputation
        X = self.extended_imputation(X, is_training=True)
        
        # Apply one-hot encoding to specified categorical columns
        if self.categorical_columns_to_encode:
            # Only encode columns that exist in the data
            cols_to_encode = [col for col in self.categorical_columns_to_encode if col in X.columns]
            if cols_to_encode:
                X = pd.get_dummies(X, columns=cols_to_encode, drop_first=True)
        
        # Store feature columns (after all transformations)
        self.feature_columns = X.columns.tolist()
        
        # Ensure all columns are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
        # Fill any NaN values that might have been introduced
        X = X.fillna(0)
        
        print(f"Training completed:")
        print(f"  - Shape after preprocessing: {X.shape}")
        print(f"  - Columns with imputation: {len(self.imputation_summary)}")
        print(f"  - Tracking columns added: {sum(1 for col in X.columns if '_was_imputed' in col)}")
        
        return X, y
    
    def get_imputation_summary(self):
        return self.imputation_summary
    

