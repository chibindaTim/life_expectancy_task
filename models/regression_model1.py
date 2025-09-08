#Model 1 draft
#L1 regularization
import numpy as np

class LassoRegression:
    def __init__(self, learning_rate=0.01, alpha=1.0,max_iter=1000 ): #alpha is the threshold strength and max_iter is number of times you can descend the slope
        
        self.learning_rate = learning_rate #intent to use gradient descent to minimize error in the linear regression model
        self.feature_names = None
        self.selected_features = {}
        self.max_iter = max_iter
        self.alpha = alpha 
    def _soft_threshold(self, x, thresh): #thresh = learning_rate*regularization_strength which is alpha
        # Handles the non-differentiability at 0
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0) #if weight is small set to 0, if weight is large shrink towards 0

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize coefficients
        self.weights = np.zeros(n_features)   # start with w=0 
        self.bias = 0                         # intercept

        # Gradient Descent Loop
        for i in range(self.max_iter):
            # Prediction get dot product of the features with weights and include self bias
            #Formula Xw+b where X is a nx d matrix (n samples d features), and w is a dx1 matrix(d weights)
            y_pred = np.dot(X, self.weights) + self.bias 

            #Compute Residual (errors) 
            residuals = y_pred - y

            #Compute gradients, which is partial derivatives , wrt to weights and bias respectively
            dw = (1/n_samples) * np.dot(X.T, residuals) #
            db = (1/n_samples) * np.sum(residuals) #

            # Gradient descent update with shrinkage
            self.weights = self._soft_threshold(
                self.weights - self.learning_rate * dw,
                self.alpha * self.learning_rate
            )
            self.bias -= self.learning_rate * db   # bias not penalized

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def get_params(self):
        """Return learned coefficients and intercept"""
        return {"weights": self.weights, "bias": self.bias}


# Separate features and target df_imputed_copy= df_imputed.copy() # Split features and target y = df_imputed_copy['Life expectancy'].astype(float).values X = df_imputed_copy.drop(columns=['Life expectancy','Country','Status']).astype(float).values lasso = LassoRegression(alpha=0.1, learning_rate=0.01, max_iter=1000) lasso.fit(X, y) print(lasso.get_params())

    