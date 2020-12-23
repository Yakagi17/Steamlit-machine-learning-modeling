'''
Importing some important library
'''
#Mian web App
import streamlit as st

#Web Chart
# from streamlit_echarts import st_echarts, st_pyecharts
# from pyecharts import options as opts
# from pyecharts.charts import Bar

#Machine Learning 
import sklearn
import pandas as pd
# from utils.file_management import cache_dataset
# import numpy as np
# import matplotlib.pyplot as plt

# #Preprocess dataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

def get_dataset():
    #Import Dataset
    dataset_df = pd.DataFrame()
    dataset_column = []
    
    csv_file_bytes = st.file_uploader("Upload a file (csv format)", type=("csv"), accept_multiple_files=False)
    
    if csv_file_bytes:
        # dataset_df = cache_dataset(csv_file_bytes)
        dataset_df = pd.read_csv(csv_file_bytes)
        dataset_name = csv_file_bytes.name.replace(".csv","")
        st.subheader(f'{dataset_name} Data set')
        dataset_column = dataset_df.columns.tolist()


        st.dataframe(dataset_df)
    else:
        st.success("Please upload dataset")
    #     st.error("Please upload proper csv format file")


    return dataset_df, dataset_column
    






    #Dataset
    # data_url = "dataset/iris.csv"
    
    # regression_df = pd.read_csv("dataset/california_housing_train.csv")
    # clustering_df = pd.read_csv("dataset/Mall_Customers.csv")

    
    

# # SPLIT_RATIO = 0.8
# processed_features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# target = ['species']

# X = classification_df[processed_features].values
# y = classification_df[target].values
# class_name = classification_df.species.unique().tolist()
# feature_names = classification_df.columns[:-1].tolist()
# X_train, X_valid, y_train, y_valid= train_test_split(X, y, test_size=SPLIT_RATIO, shuffle=False)



# Modeling
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC 
# from sklearn.neural_network import MLPClassifier

# clf_model = SVC(probability=True, random_state = 101)
# clf_model.fit(X, y)

# #Evaluation
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# result = clf_model.predict(X)
# result_proba = clf_model.predict_proba(X)
# accuracy = round(accuracy_score(y, result), 2)
# conf_matrix = confusion_matrix(y, result) #tolist()
# clf_report = classification_report(y, result, output_dict=True)



#Interpretation
'''
from sklearn.inspection import permutation_importance, partial_dependence

feature_importance = permutation_importance(clf_model, X, y, scoring='accuracy')
fi_socre = feature_importance.importances_mean.tolist()
fi_socre = [round(score, 3) for score in fi_socre]

pdp = {}
axes = {}

for i in range(len(feature_names)):
    pdp[i], axes[i] = partial_dependence(clf_model, X=classification_df[feature_names], features=feature_names[i])

'''

#Visualize

# st.title('Machine Learning Modeling - Demo')
# st.text('Demo classification modeling with SVM(Support Vector Machine) using Iris dataset')




# st.header('Evaluation :')
# st.text(f'Classification accuracy of this SVM model is : {accuracy*100} %')
# st.table(conf_matrix)
# st.table(pd.DataFrame(clf_report))

# st.header('Interpretation :')

'''
from utils.plotting import feature_importance_plot

pi_plot =  feature_importance_plot(fi_score)

st_echarts(pi_plot)

from utils.plotting import auc_plot
auc_roc_plot = auc_plot(class_name, y, result_proba)

st_echarts(auc_roc_plot)
'''








