import numpy as np
import pandas as pd
import pickle
from data_preprocessing import  load_prerocess_data
from regression_model1 import LassoRegression
from regression_model1 import RidgeRegression
import pickle

def train_and_save_models():
    # Load data
    X, y, _ = load_prerocess_data('chibindaTim/life_expectancy_task/data/Life Expectancy - Life Expectancy (1).csv')

    # Train models (example)
    model1 = LassoRegression( learning_rate=0.01, alpha=1.0,max_iter=1000)
    model1.fit(X, y)

    model2 = RidgeRegression( learning_rate=0.01, alpha=1.0,max_iter=1000)
    model2.fit(X, y)

    # Save models
    with open("models/regression_model1.pkl", "wb") as f:
        pickle.dump(model1, f)
    with open("models/regression_model2.pkl", "wb") as f:
        pickle.dump(model2, f)

    # Pick best (based on RMSE or R²)
    with open("models/regression_model_final.pkl", "wb") as f:
        pickle.dump(model1, f)  # suppose model1 is best

if __name__ == "__main__":
    train_and_save_models()
