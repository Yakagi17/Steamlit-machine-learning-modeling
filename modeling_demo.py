#Main 
import streamlit as st

#Modeling
from modeling.classification import KNN_Classifer, SVM_Classifier, MLP_Classifer
from modeling.regression import KNN_Regression, SVM_Regression, MLP_Regression
from modeling.clustering import Kmeans_Clustering

#Utilies
from utils.data_management import DataPreparation

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
        selected_features = st.sidebar.multiselect("Choose Features as predictor : ", list(dataset_column), list(dataset_column))
        if not selected_features:
            st.sidebar.error("Please select at least one feature")

        st.sidebar.write("## Method Parameter")
        k_neighbors = st.sidebar.number_input("Choose number of k-neighbors :", min_value=1, max_value=50, value=5, step=1)
        not_selected_features = [feature for feature in dataset_column if feature not in selected_features]
        target = st.sidebar.selectbox("Choose Target column : ", not_selected_features)

        is_run = st.sidebar.button("Run")
        is_feature_target =  selected_features and target

        if  is_run and is_feature_target:
            #Preprocessing
            dp = DataPreparation(dataframe=dataset_df, selected_features=selected_features, target=target)
            class_name = dp.get_class_names()
            X, y = dp.data_modeling()

            #Modeling
            clf = KNN_Classifer(k_neighbors=k_neighbors)
            y_pred, y_proba = clf.modeling(X, y)

            #Metric Evaluation
            accuracy, f1score, roc_score, conf_matrix, clf_report = clf.metircs_model(y, y_pred, class_name)

            #Interpretation
            fi_score = clf.feature_importance(X, y, selected_features)


            #Interface
            st.write('## **Metrics**')
            
            st.write('### **Main Metrics**')
            st.write(f'Accuracy : {accuracy}')
            st.write(f'F1 Score : {f1score}')
            st.write(f'AUC ROC Score : {roc_score}')

            st.write('### Confusion Matrix')
            st.dataframe(conf_matrix)
            
            st.write('### Classification Report')
            st.dataframe(clf_report)

            st.write('## **Interpretation**')
            fi_plot_options = feature_importance_plot(fi_score)
            st_echarts(fi_plot_options)
        else:
            st.sidebar.error("Please fill all method parameter")
            
    elif classification_modeling_type == "SVM(Support Vector Machine)":
        selected_features = st.sidebar.multiselect("Choose Features as predictor : ", list(dataset_column), list(dataset_column))
        if not selected_features:
            st.sidebar.error("Please select at least one feature")
        
        st.sidebar.write("## Method Parameter")

        C_range = [1,10,100,1000]
        kernel_type = ['linear','poly','rbf','sigmoid','precomputed']

        C = float(st.sidebar.selectbox("C ", C_range, index=0))
        kernel = st.sidebar.selectbox("kernel ", kernel_type)
        degree = st.sidebar.number_input("degree", min_value=1, value=3, step=1)
        gamma = st.sidebar.slider("gamma ", min_value=0.1, max_value=1.0, value=1.0, step=0.01)
        coef0 = st.sidebar.number_input("coef0", min_value=0,value=0)
        
        not_selected_features = [feature for feature in dataset_column if feature not in selected_features]
        target = st.sidebar.selectbox("Choose Target column : ", not_selected_features)

        is_run = st.sidebar.button("Run")
        is_feature_target =  selected_features and target

        if  is_run and is_feature_target:
            #Preprocessing
            dp = DataPreparation(dataframe=dataset_df, selected_features=selected_features, target=target)
            class_name = dp.get_class_names()
            X, y = dp.data_modeling()

            #Modeling
            clf = SVM_Classifier(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
            y_pred, y_proba = clf.modeling(X, y)

            #Metric Evaluation
            accuracy, f1score, roc_score, conf_matrix, clf_report = clf.metircs_model(y, y_pred, class_name)

            #Interpretation
            fi_score = clf.feature_importance(X, y, selected_features)


            #Interface
            st.write('## **Metrics**')
            
            st.write('### **Main Metrics**')
            st.write(f'Accuracy : {accuracy}')
            st.write(f'F1 Score : {f1score}')
            st.write(f'AUC ROC Score : {roc_score}')

            st.write('### Confusion Matrix')
            st.dataframe(conf_matrix)
            
            st.write('### Classification Report')
            st.dataframe(clf_report)

            st.write('## **Interpretation**')
            fi_plot_options = feature_importance_plot(fi_score)
            st_echarts(fi_plot_options)
        else:
            st.sidebar.error("Please fill all method parameter")

    elif classification_modeling_type == "MLP(Multi Layer Perceptron)":
        selected_features = st.sidebar.multiselect("Choose Features as predictor : ", list(dataset_column), list(dataset_column))
        if not selected_features:
            st.sidebar.error("Please select at least one feature")
        
        st.sidebar.write("## Method Parameter")

        C_range = [1,10,100,1000]
        activation_type = ['identity','logistic','tanh','relu']
        solver_type = ['lbfgs','sgd','adam']

        hidden_layer_str = st.sidebar.text_input("Hidden layer ", value="10,10", max_chars=20)
        hidden_layer = tuple(map(int, hidden_layer_str.split(',')))
        activation = st.sidebar.selectbox("activation function ", activation_type, index=3)
        solver = st.sidebar.selectbox("solver ", solver_type, index=2)
        alpha = st.sidebar.number_input("alpha ", value=0.0001)
        learning_rate = st.sidebar.number_input("learning rate ", value=0.001)
        max_iter = st.sidebar.number_input("maximum iteration ", value=200, min_value=1, step=1)
        early_stopping = st.sidebar.checkbox("early stopping ", value=False)
        
        not_selected_features = [feature for feature in dataset_column if feature not in selected_features]
        target = st.sidebar.selectbox("Choose Target column : ", not_selected_features)

        is_run = st.sidebar.button("Run")
        is_feature_target =  selected_features and target


        if  is_run and is_feature_target:
            #Preprocessing
            dp = DataPreparation(dataframe=dataset_df, selected_features=selected_features, target=target)
            class_name = dp.get_class_names()
            X, y = dp.data_modeling()

            #Modeling
            clf = MLP_Classifer(hidden_layer_sizes=hidden_layer, activation=activation, solver=solver, alpha=alpha, learning_rate_init=learning_rate, max_iter=max_iter, early_stopping=early_stopping)
            y_pred, y_proba = clf.modeling(X, y)

            #Metric Evaluation
            accuracy, f1score, roc_score, conf_matrix, clf_report = clf.metircs_model(y, y_pred, class_name)

            #Interpretation
            fi_score = clf.feature_importance(X, y, selected_features)


            #Interface
            st.write('## **Metrics**')
            
            st.write('### **Main Metrics**')
            st.write(f'Accuracy : {accuracy}')
            st.write(f'F1 Score : {f1score}')
            st.write(f'AUC ROC Score : {roc_score}')

            st.write('### Confusion Matrix')
            st.dataframe(conf_matrix)
            
            st.write('### Classification Report')
            st.dataframe(clf_report)

            st.write('## **Interpretation**')
            fi_plot_options = feature_importance_plot(fi_score)
            st_echarts(fi_plot_options)
        else:
            st.sidebar.error("Please fill all method parameter")


def regression_modeling(dataset_df, dataset_column):
    import streamlit as st
    REGRESSION_MOETHOD = ["KNN(K-Nearest Neighbors)", "SVM(Support Vector Machine)", "MLP(Multi Layer Perceptron)"]
    
    regression_modeling_type = st.sidebar.selectbox("Choose Regression Method :", REGRESSION_MOETHOD, 0)

    if regression_modeling_type == "KNN(K-Nearest Neighbors)":
        selected_features = st.sidebar.multiselect("Choose Features as predictor : ", list(dataset_column), list(dataset_column))
        if not selected_features:
            st.sidebar.error("Please select at least one feature")

        st.sidebar.write("## Method Parameter")
        k_neighbors = st.sidebar.number_input("Choose number of k-neighbors :", min_value=1, max_value=50, value=5, step=1)
        not_selected_features = [feature for feature in dataset_column if feature not in selected_features]
        target = st.sidebar.selectbox("Choose Target column : ", not_selected_features)

        is_run = st.sidebar.button("Run")
        is_feature_target =  selected_features and target

        if  is_run and is_feature_target:
            #Preprocessing
            dp = DataPreparation(dataframe=dataset_df, selected_features=selected_features, target=target)
            class_name = dp.get_class_names()
            X, y = dp.data_modeling()

            #Modeling
            regre = KNN_Regression(k_neighbors=k_neighbors)
            y_pred = regre.modeling(X, y)

            #Metric Evaluation
            rmse, mae, r2 = regre.metircs_model(y, y_pred, class_name)

            #Interpretation
            fi_score = regre.feature_importance(X, y, selected_features)


            #Interface
            st.write('## **Metrics**')
            
            st.write('### **Main Metrics**')
            st.write(f'RMSE Score : {rmse}')
            st.write(f'MAE Score : {mae}')
            st.write(f'R Squared Score : {r2}')

            st.write('## **Interpretation**')
            fi_plot_options = feature_importance_plot(fi_score)
            st_echarts(fi_plot_options)
        else:
            st.sidebar.error("Please fill all method parameter")
            
    elif regression_modeling_type == "SVM(Support Vector Machine)":
        selected_features = st.sidebar.multiselect("Choose Features as predictor : ", list(dataset_column), list(dataset_column))
        if not selected_features:
            st.sidebar.error("Please select at least one feature")
        
        st.sidebar.write("## Method Parameter")

        C_range = [1,10,100,1000]
        kernel_type = ['linear','poly','rbf','sigmoid','precomputed']

        C = float(st.sidebar.selectbox("C ", C_range, index=0))
        kernel = st.sidebar.selectbox("kernel ", kernel_type)
        degree = st.sidebar.number_input("degree", min_value=1, value=3, step=1)
        gamma = st.sidebar.slider("gamma ", min_value=0.1, max_value=1.0, value=1.0, step=0.01)
        coef0 = st.sidebar.number_input("coef0", min_value=0,value=0)
        
        not_selected_features = [feature for feature in dataset_column if feature not in selected_features]
        target = st.sidebar.selectbox("Choose Target column : ", not_selected_features)

        is_run = st.sidebar.button("Run")
        is_feature_target =  selected_features and target

        if  is_run and is_feature_target:
            #Preprocessing
            dp = DataPreparation(dataframe=dataset_df, selected_features=selected_features, target=target)
            class_name = dp.get_class_names()
            X, y = dp.data_modeling()

            #Modeling
            regre = SVM_Regression(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
            y_pred = regre.modeling(X, y)

            #Metric Evaluation
            rmse, mae, r2 = regre.metircs_model(y, y_pred, class_name)

            #Interpretation
            fi_score = regre.feature_importance(X, y, selected_features)


            #Interface
            st.write('## **Metrics**')
            
            st.write('### **Main Metrics**')
            st.write(f'RMSE Score : {rmse}')
            st.write(f'MAE Score : {mae}')
            st.write(f'R Squared Score : {r2}')

            st.write('## **Interpretation**')
            fi_plot_options = feature_importance_plot(fi_score)
            st_echarts(fi_plot_options)
        else:
            st.sidebar.error("Please fill all method parameter")

    elif regression_modeling_type == "MLP(Multi Layer Perceptron)":
        selected_features = st.sidebar.multiselect("Choose Features as predictor : ", list(dataset_column), list(dataset_column))
        if not selected_features:
            st.sidebar.error("Please select at least one feature")
        
        st.sidebar.write("## Method Parameter")

        C_range = [1,10,100,1000]
        activation_type = ['identity','logistic','tanh','relu']
        solver_type = ['lbfgs','sgd','adam']

        hidden_layer_str = st.sidebar.text_input("Hidden layer ", value="10,10", max_chars=20)
        hidden_layer = tuple(map(int, hidden_layer_str.split(',')))
        activation = st.sidebar.selectbox("activation function ", activation_type, index=3)
        solver = st.sidebar.selectbox("solver ", solver_type, index=2)
        alpha = st.sidebar.number_input("alpha ", value=0.0001)
        learning_rate = st.sidebar.number_input("learning rate ", value=0.001)
        max_iter = st.sidebar.number_input("maximum iteration ", value=200, min_value=1, step=1)
        early_stopping = st.sidebar.checkbox("early stopping ", value=False)
        
        not_selected_features = [feature for feature in dataset_column if feature not in selected_features]
        target = st.sidebar.selectbox("Choose Target column : ", not_selected_features)

        is_run = st.sidebar.button("Run")
        is_feature_target =  selected_features and target


        if  is_run and is_feature_target:
            #Preprocessing
            dp = DataPreparation(dataframe=dataset_df, selected_features=selected_features, target=target)
            class_name = dp.get_class_names()
            X, y = dp.data_modeling()

            #Modeling
            regre = MLP_Regression(hidden_layer_sizes=hidden_layer, activation=activation, solver=solver, alpha=alpha, learning_rate_init=learning_rate, max_iter=max_iter, early_stopping=early_stopping)
            y_pred = regre.modeling(X, y)

            #Metric Evaluation
            rmse, mae, r2 = regre.metircs_model(y, y_pred, class_name)

            #Interpretation
            fi_score = regre.feature_importance(X, y, selected_features)


            #Interface
            st.write('## **Metrics**')
            
            st.write('### **Main Metrics**')
            st.write(f'RMSE Score : {rmse}')
            st.write(f'MAE Score : {mae}')
            st.write(f'R Squared Score : {r2}')

            st.write('## **Interpretation**')
            fi_plot_options = feature_importance_plot(fi_score)
            st_echarts(fi_plot_options)
        else:
            st.sidebar.error("Please fill all method parameter")


def clustering_modeling(dataset_df, dataset_column):
    import streamlit as st
    CLUSTERING_METHOD = ["KMeans"]

    clustering_modeling_type = st.sidebar.selectbox("Choose Clustering Method :", CLUSTERING_METHOD, 0)

    if clustering_modeling_type == "KMeans":
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

