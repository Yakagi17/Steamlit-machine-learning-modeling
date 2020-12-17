#Baisic library
import numpy as np
import pandas as pd

#Preprocesing library
from sklearn.preprocessing import LabelEncoder

#Modeling library
from sklearn.cluster import KMeans

#Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve

#Interpretation
from sklearn.inspection import partial_dependence, permutation_importance

class KmeansClustering:
    def __init__(self, k_clusters=5, verbose=0, ignore_warnings=True, random_state=42):
        self.k_clusters = k_clusters
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.random_state = random_state