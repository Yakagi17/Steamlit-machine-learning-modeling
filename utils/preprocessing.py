import numpy as np
import pandas as pd

import random

from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler

#Class Target Encoding
def class_encoder(column, class_name):
    label_Encoder = LabelEncoder()
    label_Encoder.fit(class_name)
    colum = label_Encoder.transform(column)

    return colum

#Category Encoding
def categorical_encoder(dataframe, target):
    for column in dataframe.columns:
        if dataframe[column].dtype.name == 'object' and column != target:
            encode = LabelEncoder()
            dataframe[column] = encode.fit_transform(dataframe[column])
    
    return dataframe

#Remove Duplicate data
def remove_duplicate(dataframe):
    dataframe.drop_duplicates(subset=None, keep='first', inplace=True)

    return dataframe


def fill_na_data(dataframe, num_record_dataframe):
    for column in dataframe.columns:
        if dataframe[column].dtype.name != 'object':
            if (dataframe[column].isnull().sum()*2) >= num_record_dataframe:
                dataframe = dataframe.drop([column], axis=1)
            else:
                dataframe[column] = dataframe[column].fillna(dataframe[column].mean())
    
    return dataframe

#Scaling numeric data
def scaling_data(dataframe, target):
    dataframe_columns = [column for column in dataframe.columns if column != target]
    target_values = dataframe[target].values
    scaling = StandardScaler()
    dataframe = dataframe.drop([target], axis=1)
    dataframe = scaling.fit_transform(dataframe)
    dataframe = pd.DataFrame(dataframe, columns = dataframe_columns)
    dataframe[target] = target_values

    return dataframe

def oversampling(X, y):
    smote = random.choice([SMOTE(), RandomOverSampler()])
    X,y = smote.fit_resample(X,y)

    return X, y


def cleansing_data(dataframe, target):
    num_record_dataframe = dataframe.shape[0]

    dataframe = remove_duplicate(dataframe)
    dataframe = fill_na_data(dataframe, num_record_dataframe)
    dataframe = categorical_encoder(dataframe, target)
    dataframe = scaling_data(dataframe, target)

    return dataframe