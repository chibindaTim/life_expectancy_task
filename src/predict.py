import argparse
import pickle
import pandas as pd
import numpy as np

#Define Regression Metrics to evaluate model
def evaluate_model(y_val, y_pred):
    mse = np.mean((y_val-y_pred)**2) #mean squared error
    rmse = np.sqrt(mse) #root mean square error/validation error
    #find r^2 score
    #use formula: 1 - [(Residual Sum of Squares (RSS))/(Total Sum of Squares(TSS))]
    rss = np.sum((y_val-y_pred)**2)
    mean= np.mean(y_val)
    tss = np.sum((y_val-mean)**2)
    r_sqaured_score= 1-(rss/tss)

    return {'mse': mse, 'rmse': rmse, 'r_sqaured_score': r_sqaured_score}

def write_metrics_file(metrics, filepath):
    #write in req format
    with open(filepath, 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {metrics['mse']:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}\n")
        f.write(f"R-squared (R²) Score: {metrics['r2']:.2f}\n")

def save_predictions(predictions, filepath):
    """Save predictions in required format"""
    np.savetxt(filepath, predictions, delimiter=',', fmt='%.6f')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to saved model')
    parser.add_argument('--data_path', required=True, help='Path to data CSV')
    parser.add_argument('--metrics_output_path', required=True, help='Path for metrics output')
    parser.add_argument('--predictions_output_path', required=True, help='Path for predictions output')
    
    args = parser.parse_args()
    
    # Load model
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load preprocessor
    try:
        with open('../models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    except FileNotFoundError:
        preprocessor = None
    
    # Load and preprocess data
    df = pd.read_csv(args.data_path)
    
    # Extract target (assume last column or specific name)
    target_col = 'Life expectancy'  # Change per dataset
    if target_col in df.columns:
        y_true = df[target_col].values
        X = df.drop(target_col, axis=1)
    else:
        # If no target column, assume all columns are features
        X = df
        y_true = None
    
    # Preprocess features if preprocessor exists
    if preprocessor:
        X_processed = preprocessor.transform(X)
    else:
        X_processed = X.values
    
    # Make predictions
    predictions = model.predict(X_processed)
    
    # Calculate metrics if true values available
    if y_true is not None:
        metrics = evaluate_model(y_true, predictions)
        write_metrics_file(metrics, args.metrics_output_path)
        print(f"Metrics saved to {args.metrics_output_path}")
    
    # Save predictions
    save_predictions(predictions, args.predictions_output_path)
    print(f"Predictions saved to {args.predictions_output_path}")

if __name__ == "__main__":
    main()          