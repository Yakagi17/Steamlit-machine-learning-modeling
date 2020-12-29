import streamlit as st

import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self, dataframe, selected_features, target=None):
        self.dataframe = dataframe
        self.selected_features = selected_features
        self.target = target

    def data_modeling(self, split_test_size = 0.2, shuffle_data=True):
        #Shuffle
        if shuffle_data:
            self.dataframe = shuffle(self.dataframe, random_state=42)

        if self.target is not None:
            X = self.dataframe[self.selected_features]
            y = self.dataframe[self.target]

            X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size = split_test_size)

            return X_train, X_valid, y_train, y_valid
        else:
            X = self.dataframe[self.selected_features]

            return X
        

    def get_class_names(self):
        if self.target is not None:
            class_name = self.dataframe[self.target].unique().tolist()
        else:
            class_name = None

        return class_name

    def get_num_class(self):
        if self.target is not None:
            num_class = len(self.dataframe[self.target].unique().tolist())
        else:
            num_class = None

        return num_class 

    def get_features_name(self):
        return self.dataframe.columns.tolist()
    


def get_dataset():
    #Import Dataset
    dataset_df = pd.DataFrame()
    dataset_column = []
    
    csv_file_bytes = st.file_uploader("Upload a file (csv format)", type=("csv"), accept_multiple_files=False)
    
    if csv_file_bytes:
        # dataset_df = cache_dataset(csv_file_bytes)
        dataset_df = pd.read_csv(csv_file_bytes)
        dataset_name = csv_file_bytes.name.replace(".csv","")
        st.subheader(f'Raw {dataset_name} Data set')
        dataset_column = dataset_df.columns.tolist()


        st.dataframe(dataset_df)
    else:
        st.success("Please upload dataset")
    #     st.error("Please upload proper csv format file")


    return dataset_df, dataset_column