from models.regression_model1 import LassoRegression
import pamdas as pd
#since the dataset has missing values use extended imputation for data preprocessing
def imputation(df):
    # Create a copy 
    df_imputed = df.copy()
    
    # Get columns with missing values
    missing_columns = df.columns[df.isnull().any()].tolist()
    
    imputation_summary = {}
    
    for col in missing_columns:
        # Count original missing values
        original_missing = df[col].isnull().sum()
        missing_percentage = (original_missing / len(df)) * 100 #check the impact the column has if it were dropped because of NaN values
        
        # Create tracking column (True where values were missing)
        tracking_col_name = f"{col}_was_imputed"
        df_imputed[tracking_col_name] = df[col].isnull().astype(int) #return 0 for true and 1 for false intead of true false
        
        # Determine imputation strategy based on data type
        if df[col].dtype == 'object':
            # Categorical data 
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                fill_value = mode_value[0]
                method = 'mode'
            else:
                fill_value = 'N/A'
                method = 'default_string'
        
        else: #df[col].dtype in ['int64', 'float64']:
            # Numerical data 
            fill_value = df[col].median()
            method = 'median'
    
        # Apply imputation
        df_imputed[col] = df_imputed[col].fillna(fill_value)
        
        # Verify imputation worked
        remaining_missing = df_imputed[col].isnull().sum()
        imputed_count = original_missing - remaining_missing
        
        # Store summary
        imputation_summary[col] = {
            'original_missing': original_missing,
            'missing_percentage': missing_percentage,
            'method': method,
            'fill_value': fill_value,
            'imputed_count': imputed_count,
            'tracking_column': tracking_col_name
        }

        #Copy
        df_copy= df_imputed.copy()

        # One-hot encode categorical columns (Country, Status) , when working without them rsme was >4 but <5
        df_encoded = pd.get_dummies(df_copy, columns=['Country', 'Status'], drop_first=True)#.astype(int)


    return df_encoded, imputation_summary
df_encoded, imputation_summary = imputation(df) # type: ignore

df_encoded.shape
df_encoded.head()

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Separate target
y = df_encoded['Life expectancy'].astype(float).values

# Drop object columns and target
X = df_encoded.drop(columns=['Life expectancy']).astype(float).values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Fit Lasso ---
lasso = LassoRegression(alpha=0.001, learning_rate=0.001, max_iter=1000)  # also lowered LR
lasso.fit(X_scaled, y)

#print(lasso.get_params())