#Main 
import streamlit as st

#Modeling
from modeling.clustering import Kmeans_Clustering

#Utilies
from utils.data_management import DataPreparation
from utils.preprocessing import cleansing_data
# from utils.preprocessing import remove_duplicate, categorical_encoder, class_encoder, fill_na_data, scaling_data, oversampling

#Plotting
from utils.plotting import feature_importance_plot, auc_plot, partial_dependence_plot, coordinates_plot
from streamlit_echarts import st_echarts



def kmeans_clustering_interface(dataset_df, dataset_column):
    selected_features = st.sidebar.multiselect("Choose Features as predictor : ", list(dataset_column), list(dataset_column))
    if not selected_features:
        st.sidebar.error("Please select at least one feature")

    st.sidebar.write("## Method Parameter")
    k_clusters = st.sidebar.number_input("Choose number of K :", min_value=1, max_value=50, value=5, step=1)

    is_run = st.sidebar.button("Run")

    if  is_run and selected_features:
        #Preprocessing
        dp = DataPreparation(dataframe=dataset_df, selected_features=selected_features)
        X = dp.data_modeling()

        #Modeling
        clus = Kmeans_Clustering(k_clusters=k_clusters)
        cluster_pred = clus.modeling(X)

        #Metric Evaluation
        distance, centroids, cluster_label = clus.metircs_model(X)

        #Interpretation
        centroid_df, clusteer_data_point_df = clus.interpretation(X, centroids, cluster_pred)

        #Interface
        st.write('## **Metrics**')
        
        st.write('### **Main Metrics**')
        # st.write(f'Accuracy : {}')
        # st.write(f'F1 Score : {}')
        # st.write(f'AUC ROC Score : {}')

        st.write('## **Interpretation**')
        #Centroid Coordinate Plot
        centroid_options = coordinates_plot(centroid_df, selected_features, num_cluster=k_clusters)
        st_echarts(centroid_options)

        #Cluster data Coordinate Plot
        cluster_data_options = coordinates_plot(clusteer_data_point_df, selected_features, num_cluster=k_clusters)
        st_echarts(cluster_data_options)

    else:
        st.sidebar.error("Please fill all method parameter")
