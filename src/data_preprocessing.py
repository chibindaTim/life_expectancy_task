import pandas as pd
import numpy as np
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


        # One-hot encode categorical columns (Country, Status) , when working without them rsme was >4 but <5
    df_encoded = pd.get_dummies(df_imputed, columns=['Country', 'Status'], drop_first=True)#.astype(int)


    return df_encoded, imputation_summary

#Load and prerocess data
def load_prerocess_data(data_path):
    df = pd.read_csv(data_path)
    df_encoded, summary = imputation(df)

    # Strip column names (in case of trailing spaces in CSV)
    df_encoded.columns = df_encoded.columns.str.strip()

    target='Life expectancy'
    y=df[target]
    X=df.drop(target, axis =1)

    return X, y, summary    
    
