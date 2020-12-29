#Main 
import streamlit as st

#Modeling
from interface.classification_interface import knn_classification_interface, svm_classfification_interface, mlp_classification_interface
from interface.regression_interface import knn_regression_interface, svm_regression_interface, mlp_regression_interface
from interface.clustering_interface import kmeans_clustering_interface

#Utilies
from utils.data_management import DataPreparation
# from processing_data import cleansing_data
# from utils.preprocessing import remove_duplicate, categorical_encoder, class_encoder, fill_na_data, scaling_data, oversampling

#Plotting
from utils.plotting import feature_importance_plot, auc_plot, partial_dependence_plot, coordinates_plot
from streamlit_echarts import st_echarts


def startup():
    import streamlit as st

    st.sidebar.success("Select type of modeling above.")



def classification_modeling(dataset_df, dataset_column):
    import streamlit as st
    CLASSIFICATION_MOETHOD = ["KNN(K-Nearest Neighbors)", "SVM(Support Vector Machine)", "MLP(Multi Layer Perceptron)"]


    classification_modeling_type = st.sidebar.selectbox("Choose Classification Method :", CLASSIFICATION_MOETHOD, 0)

    if classification_modeling_type == "KNN(K-Nearest Neighbors)":
        knn_classification_interface(dataset_df, dataset_column)

    elif classification_modeling_type == "SVM(Support Vector Machine)":
        svm_classfification_interface(dataset_df, dataset_column)

    elif classification_modeling_type == "MLP(Multi Layer Perceptron)":
        mlp_classification_interface(dataset_df, dataset_column)

def regression_modeling(dataset_df, dataset_column):
    import streamlit as st
    REGRESSION_MOETHOD = ["KNN(K-Nearest Neighbors)", "SVM(Support Vector Machine)", "MLP(Multi Layer Perceptron)"]
    
    regression_modeling_type = st.sidebar.selectbox("Choose Regression Method :", REGRESSION_MOETHOD, 0)

    if regression_modeling_type == "KNN(K-Nearest Neighbors)":
        knn_regression_interface(dataset_df, dataset_column)
            
    elif regression_modeling_type == "SVM(Support Vector Machine)":
        svm_regression_interface(dataset_df, dataset_column)

    elif regression_modeling_type == "MLP(Multi Layer Perceptron)":
        mlp_regression_interface(dataset_df, dataset_column)

def clustering_modeling(dataset_df, dataset_column):
    import streamlit as st
    CLUSTERING_METHOD = ["KMeans"]

    clustering_modeling_type = st.sidebar.selectbox("Choose Clustering Method :", CLUSTERING_METHOD, 0)

    if clustering_modeling_type == "KMeans":
        kmeans_clustering_interface(dataset_df, dataset_column)