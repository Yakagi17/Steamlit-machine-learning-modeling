#Baisic library
import numpy as np
import pandas as pd

#Preprocesing library
from sklearn.preprocessing import LabelEncoder

#Modeling library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier

#Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score

# #Interpretation
from sklearn.inspection import partial_dependence, permutation_importance

#Utils
from utils.preprocessing import classEncoder


class KNNClassifer:
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors=self.k_neighbors)
    
    def modeling(self, X, y):
        self.classifier.fit(X, y)
        
        y_pred = self.classifier.predict(X)
        y_proba = self.classifier.predict_proba(X)

        return y_pred, y_proba
    
    def metircs_model(self, y, y_pred, class_name):
        #Model Accuracy
        accuracy = round(accuracy_score(y, y_pred), 2)

        #Confusion Matrix
        conf_matrix = confusion_matrix(y, y_pred) #tolist()

        index = [("Predicted", index_class) for index_class in class_name]
        columns = [("True", index_class) for index_class in class_name]

        conf_matrix = pd.DataFrame(conf_matrix, columns=columns, index=index)
        conf_matrix.columns = pd.MultiIndex.from_tuples(conf_matrix.columns)

        #Classification Report
        clf_report = classification_report(y, y_pred, output_dict=True)

        #F1 Score
        if len(class_name) > 2:
            f1score = round(f1_score(y, y_pred, average='weighted'), 2)
        else:
            f1score = round(f1_score(y, y_pred, average='binary'),2)

        roc_score = 0
        # #AUC ROC Score
        # y_true = np.array(y)
        # if y_true.dtype == 'O':
        #     y_true = classEncoder(y_true, class_name)

        # roc_score = round(roc_auc_score(y_true, y_pred), 2)  
        

        #Mapping Confusion Matrix


        return accuracy, f1score, roc_score, conf_matrix, clf_report

    def feature_importance(self, X, y, features_name):
        #Calculate permutation importance score for each feature
        fi = permutation_importance(self.classifier, X, y, scoring='accuracy')
        fi_socre = fi.importances_mean.tolist()
        fi_socre = [round(score, 3) for score in fi_socre]

        #Zip and sort feature importance
        zip_fi = zip(features_name, fi_socre)
        fi_socre = sorted(zip_fi, key=lambda x:x[1], reverse=False)

        return fi_socre
    
class SVMClassifer:
    def __init__(self, C=1, kernel='rbf', degree=1, gamma=1.0, coef0=0.0):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0= coef0
        self.classifier = SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0, probability=True)
    
    def modeling(self, X, y):
        self.classifier.fit(X, y)
        
        y_pred = self.classifier.predict(X)
        y_proba = self.classifier.predict_proba(X)

        return y_pred, y_proba
    
    def metircs_model(self, y, y_pred):
        accuracy = round(accuracy_score(y, y_pred), 2)
        conf_matrix = confusion_matrix(y, y_pred) #tolist()
        clf_report = classification_report(y, y_pred, output_dict=True)
        
        return accuracy, conf_matrix, clf_report

class MLPClassifer:
    def __init__(self, hidden_layer_sizes=(100,) ,activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200, early_stopping=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.classifier = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver, alpha=self.alpha, learning_rate_init=self.learning_rate_init, max_iter=self.max_iter, early_stopping=self.early_stopping)
    
    def modeling(self, X, y):
        self.classifier.fit(X, y)
        
        y_pred = self.classifier.predict(X)
        y_proba = self.classifier.predict_proba(X)

        return y_pred, y_proba
    
    def metircs_model(self, y, y_pred):
        accuracy = round(accuracy_score(y, y_pred), 2)
        conf_matrix = confusion_matrix(y, y_pred) #tolist()
        clf_report = classification_report(y, y_pred, output_dict=True)
        
        return accuracy, conf_matrix, clf_report