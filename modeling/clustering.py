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

class Kmeans_Clustering:
    def __init__(self, k_clusters=5, verbose=0, ignore_warnings=True, random_state=42):
        self.k_clusters = k_clusters
        self.clustering = KMeans(n_clusters=k_clusters)
    
    def modeling(self, X):
        self.clustering.fit(X)
        
        y_pred = self.clustering.predict(X)

        return y_pred
    
    def metircs_model(self, X):
        distance = self.clustering.transform(X)
        centroids = self.clustering.cluster_centers_
        cluster_label = self.clustering.labels_

        return distance, centroids, cluster_label

    def interpretation(self, X, centroids, cluster_pred):
        #Centroid Coordinate
        centroid_df = pd.DataFrame(centroids, columns=X.columns)
        centroid_df['cluster'] = centroid_df.index

        #Cluster Data Point Coordinate
        cluster_data_point_df = pd.DataFrame(X, index=X.index, columns=X.columns)
        cluster_data_point_df['cluster'] = cluster_pred

        return centroid_df, cluster_data_point_df