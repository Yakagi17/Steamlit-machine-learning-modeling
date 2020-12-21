import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self, dataframe, selected_features, target):
        self.dataframe = dataframe
        self.selected_features = selected_features
        self.target = target

    def data_modeling(self, shuffle_data=True):
        #Shuffle
        if shuffle_data:
            self.dataframe = shuffle(self.dataframe, random_state=42)

        X = self.dataframe[self.selected_features]
        y = self.dataframe[self.target]

        # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=42)
        # return X_train, X_valid, y_train, y_valid

        return X, y

    def get_class_names(self):
        return self.dataframe[self.target].unique().tolist()
    
    def get_num_class(self):
        return len(self.dataframe[self.target].unique().tolist())

    def get_features_name(self):
        return self.dataframe.columns.tolist()