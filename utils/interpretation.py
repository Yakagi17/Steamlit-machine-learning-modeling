# import pandas as pd
# import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import partial_dependence, permutation_importance

def auc_score(class_name, y_true, y_proba):
    fpr = {}
    tpr = {}
    thresh = {}
    auc_roc = {}
    auc_score = []
    num_class = len(class_name)
    

    #Check if y value is String or Object type
    if y_true.dtype == 'O':
        label_Encoder = LabelEncoder()
        label_Encoder.fit(class_name)
        y_true = label_Encoder.transform(y_true)

    #Calcluate true positif rate and false positif rate
    for i in range(num_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_true, y_proba[:,i], pos_label=i)
        auc_roc[i] = (fpr[i], tpr[i]) 

    #store tpr and fpr to temp variable for reshaping data dimentions    
    for i in range(num_class):
        temp = []
        for j in range(len(fpr[i])):
            temp.append([fpr[i][j], tpr[i][j]])
        auc_score.append(temp)
    
    return auc_score, auc_roc


def feature_importance(clf_model, X, y, features    _name):
    #Calculate permutation importance score for each feature
    fi = permutation_importance(clf_model, X, y, scoring='accuracy')
    fi_socre = fi.importances_mean.tolist()
    fi_socre = [round(score, 3) for score in fi_socre]

    zip_fi = zip(features_name, fi_socre)
    fi_socre = sorted(zip_fi, key=lambda x:x[1], reverse=False)

    return fi_socre


def partial_dependence_plot(clf_model, classification_df, features_names):
    pdp = {}
    axes = {}

    for i in range(len(feature_names)):
        pdp[i], axes[i] = partial_dependence(clf_model, X=classification_df[features_names], features=features_names[i])

    return pdp, axes