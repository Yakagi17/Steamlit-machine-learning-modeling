#Main 
import streamlit as st

#Modeling
from modeling.classification import KNNClassifer

#Utilies
from utils.data_management import DataPreparation

#Plotting
from utils.plotting import feature_importance_plot, auc_plot, partial_dependence_plot
from streamlit_echarts import st_echarts
def startup():
    import streamlit as st

    st.sidebar.success("Select type of modeling above.")

def classification_modeling(dataset_df, dataset_column):
    import streamlit as st
    CLASSIFICATION_MOETHOD = ["KNN(K-Nearest Neighbors)", "SVM(Support Vector Machine)", "MLP(Multi Layer Perceptron)"]


    classification_modeling_type = st.sidebar.selectbox("Choose Classification Method :", CLASSIFICATION_MOETHOD, 0)

    if classification_modeling_type == "KNN(K-Nearest Neighbors)":
        selected_features = st.sidebar.multiselect("Choose Features as predictor : ", list(dataset_column), list(dataset_column))
        if not selected_features:
            st.sidebar.error("Please select at least one feature")

        k_neighbors = st.sidebar.number_input("Choose number of k-neighbors :", min_value=1, max_value=50, value=5, step=1)
        not_selected_features = [feature for feature in dataset_column if feature not in selected_features]
        target = st.sidebar.selectbox("Choose Target column : ", not_selected_features)

        if st.sidebar.button("Run"):
            #Preprocessing
            dp = DataPreparation(dataframe=dataset_df, selected_features=selected_features, target=target)
            class_name = dp.get_class_names()
            X, y = dp.data_modeling()

            #Modeling
            clf = KNNClassifer(k_neighbors=k_neighbors)
            y_pred, y_proba = clf.modeling(X, y)

            #Metric Evaluation
            accuracy, f1score, roc_score, conf_matrix, clf_report = clf.metircs_model(y, y_pred, class_name)

            #Interpretation
            fi_score = clf.feature_importance(X, y, selected_features)


            #Interface
            st.write('## **Metrics**')
            
            st.write('### **Mian Metrics**')
            st.write(f'Accuracy : {accuracy}')
            st.write(f'F1 Score : {f1score}')
            st.write(f'AUC ROC Score : {f1score}')

            st.write('### Confusion Matrix')
            st.write(conf_matrix)
            
            st.write('### Classification Report')
            st.dataframe(clf_report)

            st.write('## **Interpretation**')
            fi_plot_options = feature_importance_plot(fi_score)
            st_echarts(fi_plot_options)


def regression_modeling(dataset_df, dataset_column):
    import streamlit as st
    REGRESSION_MOETHOD = ["KNN(K-Nearest Neighbors)", "SVM(Support Vector Machine)", "MLP(Multi Layer Perceptron)"]
    
    regression_modeling_type = st.sidebar.selectbox("Choose Regression Method :", REGRESSION_MOETHOD, 0)

def clustering_modeling(dataset_df, dataset_column):
    import streamlit as st
    CLUSTERING_METHOD = ["KMeans"]

    clustering_modeling_type = st.sidebar.selectbox("Choose Clustering Method :", CLUSTERING_METHOD, 0)
