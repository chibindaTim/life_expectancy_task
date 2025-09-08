from models.regression_model1 import LassoRegression
import numpy as np
from src.data_preprocessing import X_scaled
#Cross Validation (Evaluation of model)
def cross_validate(X , y, alphas , cv ): #training dataset features, target , regularization strength, buckets/folds

    #cross validate with different alpha
    best_alpha = None
    best_score = float('inf')
    cv_results = {}
   
    for alpha in alphas:
        fold_size = len(X) // cv
        scores = []

    #create validation/train set
        for fold in range(cv):
        #split the dataset into train and test sets
            start = fold * fold_size #creates starting point for each fold
            if fold < cv-1:
                end=start + fold_size
            else:
                end=len(X)
    #Create the validation dataset /test dataset
            X_val =X[start:end]
            y_val =y[start:end]

    #Create training dataset , concatenate elements before start and after end
            X_train = np.concatenate([X[:start],X[end:]])
            y_train = np.concatenate([y[:start],y[end:]])

    #Train model using lasso
            LassoTrain= LassoRegression(alpha=alpha)
            LassoTrain.fit(X_train,y_train)


    #Predict and Evaluate the model
            y_pred=LassoTrain.predict(X_val)
            metrics = evaluate_model(y_val, y_pred)   # call the function
            mse = metrics['mse']                      # then access 'mse' from the result
    #returns regression metrics Regression Metrics:
    #Mean Squared Error (MSE): <value>
    #Root Mean Squared Error (RMSE): <value>
    #R-squared (R²) Score: <value>
            scores.append(mse)

        avg_score = np.mean(scores)
        std_score = np.std(scores)

        cv_results[alpha] = {
            'mean_mse': avg_score,
            'std_mse': std_score,
            'scores': scores
        }

    #select the best alpha/ learning rate
        if avg_score < best_score:
            best_score = avg_score
            best_alpha = alpha

    return best_alpha, cv_results


#alphas = [0.001,0.0001,0.5,0.55 ,0.01, 0.1, 1.0, 10.0]

#best_alpha, cv_results = cross_validate(X_scaled, y, alphas, 10)
#print("Best alpha:", best_alpha)
#print("CV results:", cv_results)

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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42 # type: ignore
) 

alphas = [0.001,0.0001,0.5,0.55 ,0.01, 0.1, 1.0, 10.0]

best_alpha, cv_results = cross_validate(X_train, y_train, alphas, cv=10)
print("Best alpha from CV:", best_alpha)
final_model = LassoRegression(alpha=best_alpha)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
metrics = evaluate_model(y_test, y_pred)

print("Test Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")
#the rmse reduces from 4 to 2 on including country and status

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("First 5 feature rows (X_test):\n", X_test[:5])
print("First 5 actual life expectancies (y_test):\n", y_test[:5])

#print("X_test shape:", X_train.shape)
#print("y_test shape:", y_train.shape)

#print("First 5 feature rows (X_train):\n", X_train[:5])
#print("First 5 actual life expectancies (y_train):\n", y_train[:5])

# Example: take first row from test set
new_data_point = X_test[:10]   # a 1D array of features

# Reshape to 2D since model expects multiple samples (n_samples, n_features)
new_data_point = new_data_point.reshape(10, -10)

# Predict life expectancy
predicted_life_exp = final_model.predict(new_data_point)
print("Predicted Life Expectancy:", predicted_life_exp[:10])


