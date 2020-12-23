#Baisic library
import numpy as np
import pandas as pd

#Preprocesing library
from sklearn.preprocessing import LabelEncoder

#Modeling library
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

#Evaluation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# #Interpretation
from sklearn.inspection import partial_dependence, permutation_importance

#Utils
from utils.preprocessing import classEncoder


class KNN_Regression:
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors
        self.regression = KNeighborsRegressor(n_neighbors=self.k_neighbors)
    
    def modeling(self, X, y):
        self.regression.fit(X, y)
        
        y_pred = self.regression.predict(X)

        return y_pred
    
    def metircs_model(self, y, y_pred, class_name):
        #RMSE
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        #MAE
        mae = mean_absolute_error(y, y_pred)

        #R Squared
        r2 = r2_score(y, y_pred)

        return rmse, mae, r2

    def feature_importance(self, X, y, features_name):
        #Calculate permutation importance score for each feature
        fi = permutation_importance(self.regression, X, y)
        fi_socre = fi.importances_mean.tolist()
        fi_socre = [round(score, 3) for score in fi_socre]

        #Zip and sort feature importance
        zip_fi = zip(features_name, fi_socre)
        fi_socre = sorted(zip_fi, key=lambda x:x[1], reverse=False)

        return fi_socre
    
class SVM_Regression:
    def __init__(self, C=1, kernel='rbf', degree=1, gamma=1.0, coef0=0.0):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0= coef0
        self.regression = SVR(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
    
    def modeling(self, X, y):
        self.regression.fit(X, y)
        
        y_pred = self.regression.predict(X)

        return y_pred
    
    def metircs_model(self, y, y_pred, class_name):
        #RMSE
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        #MAE
        mae = mean_absolute_error(y, y_pred)

        #R Squared
        r2 = r2_score(y, y_pred)

        return rmse, mae, r2

    def feature_importance(self, X, y, features_name):
        #Calculate permutation importance score for each feature
        fi = permutation_importance(self.regression, X, y)
        fi_socre = fi.importances_mean.tolist()
        fi_socre = [round(score, 3) for score in fi_socre]

        #Zip and sort feature importance
        zip_fi = zip(features_name, fi_socre)
        fi_socre = sorted(zip_fi, key=lambda x:x[1], reverse=False)

        return fi_socre

class MLP_Regression:
    def __init__(self, hidden_layer_sizes=(100,) ,activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200, early_stopping=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.regression = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver, alpha=self.alpha, learning_rate_init=self.learning_rate_init, max_iter=self.max_iter, early_stopping=self.early_stopping)
    
    def modeling(self, X, y):
        self.regression.fit(X, y)
        
        y_pred = self.regression.predict(X)

        return y_pred
    
    def metircs_model(self, y, y_pred, class_name):
        #RMSE
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        #MAE
        mae = mean_absolute_error(y, y_pred)

        #R Squared
        r2 = r2_score(y, y_pred)

        return rmse, mae, r2

    def feature_importance(self, X, y, features_name):
        #Calculate permutation importance score for each feature
        fi = permutation_importance(self.regression, X, y)
        fi_socre = fi.importances_mean.tolist()
        fi_socre = [round(score, 3) for score in fi_socre]

        #Zip and sort feature importance
        zip_fi = zip(features_name, fi_socre)
        fi_socre = sorted(zip_fi, key=lambda x:x[1], reverse=False)

        return fi_socre
