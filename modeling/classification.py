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
from utils.preprocessing import class_encoder


class KNN_Classifer:
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors=self.k_neighbors)
    
    def modeling(self, X_train, X_valid, y_train):
        self.classifier.fit(X_train, y_train)
        
        y_pred = self.classifier.predict(X_valid)
        y_proba = self.classifier.predict_proba(X_valid)

        return y_pred, y_proba
    
    def metircs_model(self, y_valid, y_pred, class_name):
        num_class = len(class_name)

        #Model Accuracy
        accuracy = round(accuracy_score(y_valid, y_pred), 4)

        #Confusion Matrix
        conf_matrix = confusion_matrix(y_valid, y_pred) #tolist()

        conf_matrix = pd.DataFrame(conf_matrix, columns=[["True"]*num_class, class_name], index=[["Predicted"]*num_class, class_name])
        # # conf_matrix.columns = pd.MultiIndex.from_tuples(conf_matrix.columns)

        #Classification Report
        clf_report = classification_report(y_valid, y_pred, output_dict=True)

        #F1 Score
        if num_class > 2:
            f1score = round(f1_score(y_valid, y_pred, average='weighted'), 4)
        else:
            f1score = round(f1_score(y_valid, y_pred, average='binary'), 4)

  
        # AUC ROC Score
        y_true = np.array(y_valid)
        y_pred = np.array(y_pred)
        
        #Check if y_true type is object
        if y_true.dtype == 'O':
            y_true = class_encoder(y_true, class_name)

        if y_pred.dtype == 'O':
            y_pred = class_encoder(y_pred, class_name)

        try:
            roc_score = round(roc_auc_score(y_true, y_pred), 4)  
        except:
            roc_score = ("Error, ROC AUC Score couldn't be calculated")
        # Mapping Confusion Matrix


        return accuracy, f1score, roc_score, conf_matrix, clf_report

    def feature_importance(self, X_valid, y_valid, features_name):
        #Calculate permutation importance score for each feature
        fi = permutation_importance(self.classifier, X_valid, y_valid, scoring='accuracy')
        fi_socre = fi.importances_mean.tolist()
        fi_socre = [round(score, 3) for score in fi_socre]

        #Zip and sort feature importance
        zip_fi = zip(features_name, fi_socre)
        fi_socre = sorted(zip_fi, key=lambda x:x[1], reverse=False)

        return fi_socre
    
class SVM_Classifier:
    def __init__(self, C=1, kernel='rbf', degree=1, gamma=1.0, coef0=0.0):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0= coef0
        self.classifier = SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0, probability=True)
    
    def modeling(self, X_train, X_valid, y_train):
        self.classifier.fit(X_train, y_train)
        
        y_pred = self.classifier.predict(X_valid)
        y_proba = self.classifier.predict_proba(X_valid)

        return y_pred, y_proba
    
    def metircs_model(self, y_valid, y_pred, class_name):
        num_class = len(class_name)

        #Model Accuracy
        accuracy = round(accuracy_score(y_valid, y_pred), 4)

        #Confusion Matrix
        conf_matrix = confusion_matrix(y_valid, y_pred) #tolist()

        # index = [("Predicted", index_class) for index_class in class_name]
        # columns = [("True", index_class) for index_class in class_name]

        conf_matrix = pd.DataFrame(conf_matrix, columns=[["True"]*num_class, class_name], index=[["Predicted"]*num_class, class_name])
        # conf_matrix.columns = pd.MultiIndex.from_tuples(conf_matrix.columns)

        #Classification Report
        clf_report = classification_report(y_valid, y_pred, output_dict=True)

        #F1 Score
        if len(class_name) > 2:
            f1score = round(f1_score(y_valid, y_pred, average='weighted'), 4)
        else:
            f1score = round(f1_score(y_valid, y_pred, average='binary'), 4)

  
        # AUC ROC Score
        y_true = np.array(y_valid)
        y_pred = np.array(y_pred)
        
        #Check if y_true type is object
        if y_true.dtype == 'O':
            y_true = class_encoder(y_true, class_name)

        if y_pred.dtype == 'O':
            y_pred = class_encoder(y_pred, class_name)

        try:
            roc_score = round(roc_auc_score(y_true, y_pred), 4)  
        except:
            roc_score = ("Error, ROC AUC Score couldn't be calculated")
        # Mapping Confusion Matrix


        return accuracy, f1score, roc_score, conf_matrix, clf_report

    def feature_importance(self, X_valid, y_valid, features_name):
        #Calculate permutation importance score for each feature
        fi = permutation_importance(self.classifier, X_valid, y_valid, scoring='accuracy')
        fi_socre = fi.importances_mean.tolist()
        fi_socre = [round(score, 3) for score in fi_socre]

        #Zip and sort feature importance
        zip_fi = zip(features_name, fi_socre)
        fi_socre = sorted(zip_fi, key=lambda x:x[1], reverse=False)

        return fi_socre


class MLP_Classifer:
    def __init__(self, hidden_layer_sizes=(100,) ,activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200, early_stopping=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.classifier = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver, alpha=self.alpha, learning_rate_init=self.learning_rate_init, max_iter=self.max_iter, early_stopping=self.early_stopping)
    
    def modeling(self, X_train, X_valid, y_train):
        self.classifier.fit(X_train, y_train)
        
        y_pred = self.classifier.predict(X_valid)
        y_proba = self.classifier.predict_proba(X_valid)

        return y_pred, y_proba
    
    def metircs_model(self, y_valid, y_pred, class_name):
        num_class = len(class_name)

        #Model Accuracy
        accuracy = round(accuracy_score(y_valid, y_pred), 4)

        #Confusion Matrix
        conf_matrix = confusion_matrix(y_valid, y_pred) #tolist()

        # index = [("Predicted", index_class) for index_class in class_name]
        # columns = [("True", index_class) for index_class in class_name]

        conf_matrix = pd.DataFrame(conf_matrix, columns=[["True"]*num_class, class_name], index=[["Predicted"]*num_class, class_name])
        # conf_matrix.columns = pd.MultiIndex.from_tuples(conf_matrix.columns)

        #Classification Report
        clf_report = classification_report(y_valid, y_pred, output_dict=True)

        #F1 Score
        if len(class_name) > 2:
            f1score = round(f1_score(y_valid, y_pred, average='weighted'), 4)
        else:
            f1score = round(f1_score(y_valid, y_pred, average='binary'), 4)

  
        # AUC ROC Score
        y_true = np.array(y_valid)
        y_pred = np.array(y_pred)
        
        #Check if y_true type is object
        if y_true.dtype == 'O':
            y_true = class_encoder(y_true, class_name)

        if y_pred.dtype == 'O':
            y_pred = class_encoder(y_pred, class_name)

        try:
            roc_score = round(roc_auc_score(y_true, y_pred), 4)  
        except:
            roc_score = ("Error, ROC AUC Score couldn't be calculated")
        # Mapping Confusion Matrix


        return accuracy, f1score, roc_score, conf_matrix, clf_report

    def feature_importance(self, X_valid, y_valid, features_name):
        #Calculate permutation importance score for each feature
        fi = permutation_importance(self.classifier, X_valid, y_valid, scoring='accuracy')
        fi_socre = fi.importances_mean.tolist()
        fi_socre = [round(score, 3) for score in fi_socre]

        #Zip and sort feature importance
        zip_fi = zip(features_name, fi_socre)
        fi_socre = sorted(zip_fi, key=lambda x:x[1], reverse=False)

        return fi_socre
