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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve

#Interpretation
from sklearn.inspection import partial_dependence, permutation_importance


# def kNN_Classifer(dataframe, selected_features, target, k_neighbors):
#     #Declare dependency variables
#     X = dataframe[selected_features].values
#     y = dataframe[target].values
#     class_names = X.unique().tolist()
#     num_class = len(class_names)

#     #Modeling
#     clf = KNeighborsClassifier(n_neighbors=k_neighbors)
#     clf.fit(X, y)

#     #Evaluation
#     result = clf.predict(X)
#     result_proba = clf.predict_proba(X)
#     accuracy = round(accuracy_score(y, result), 2)
#     conf_matrix = confusion_matrix(y, result) #tolist()
#     clf_report = classification_report(y, result, output_dict=True)




class KNNClassifer:
    def __init__(self, k_neighbors=5, verbose=0, ignore_warnings=True, random_state=42):
        self.k_neighbors = k_neighbors
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.random_state = random_state
    
    def modeling(self, dataframe, selected_features, target):
        X = dataframe[selected_features].values
        y = dataframe[target].values
        class_names = self.X.unique().tolist()
        num_class = len(self.class_names)

        clf_model = KNeighborsClassifier(n_neighbors = self.k_neighbors)
        clf_model.fit(X, y)
        
        result = clf_model.predict(X)
        result_proba = clf_model.predict_proba(X)
        accuracy = round(accuracy_score(y, result), 2)
        conf_matrix = confusion_matrix(y, result) #tolist()
        clf_report = classification_report(y, result, output_dict=True)
    
    # def set_data(self, dataframe, selected_features, target):
    #     self.selected_features = selected_features
    #     self.target = target
    #     self.X = dataframe[selected_features].values
    #     self.y = dataframe[target].values
    #     self.class_names = self.X.unique().tolist()
    #     self.num_class = len(self.class_names)