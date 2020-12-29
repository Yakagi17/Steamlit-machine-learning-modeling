#
import streamlit as st

#Modeling
from modeling.regression import KNN_Regression, SVM_Regression, MLP_Regression

#Utilies
from utils.data_management import DataPreparation
from utils.preprocessing import cleansing_data
# from utils.preprocessing import remove_duplicate, categorical_encoder, class_encoder, fill_na_data, scaling_data, oversampling

#Plotting
from utils.plotting import feature_importance_plot, auc_plot, partial_dependence_plot, coordinates_plot
from streamlit_echarts import st_echarts


def knn_regression_interface(dataset_df, dataset_column):
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
        X_train, X_valid, y_train, y_valid = dp.data_modeling()

        #Modeling
        regre = KNN_Regression(k_neighbors=k_neighbors)
        y_pred = regre.modeling(X_train, X_valid, y_train)

        #Metric Evaluation
        rmse, mae, r2 = regre.metircs_model(y_valid, y_pred, class_name)

        #Interpretation
        fi_score = regre.feature_importance(X_valid, y_valid, selected_features)


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


def svm_regression_interface(dataset_df, dataset_column):
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
        X_train, X_valid,y_train, y_valid = dp.data_modeling()

        #Modeling
        regre = SVM_Regression(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
        y_pred = regre.modeling(X_train, X_valid, y_train)

        #Metric Evaluation
        rmse, mae, r2 = regre.metircs_model(y_valid, y_pred, class_name)

        #Interpretation
        fi_score = regre.feature_importance(X_valid, y_valid, selected_features)


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


def mlp_regression_interface(dataset_df, dataset_column):
    selected_features = st.sidebar.multiselect("Choose Features as predictor : ", list(dataset_column), list(dataset_column))
    if not selected_features:
        st.sidebar.error("Please select at least one feature")
    
    st.sidebar.write("## Method Parameter")

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
        X_train, X_valid, y_train, y_valid = dp.data_modeling()

        #Modeling
        regre = MLP_Regression(hidden_layer_sizes=hidden_layer, activation=activation, solver=solver, alpha=alpha, learning_rate_init=learning_rate, max_iter=max_iter, early_stopping=early_stopping)
        y_pred = regre.modeling(X_train, X_valid,y_train, y_valid)

        #Metric Evaluation
        rmse, mae, r2 = regre.metircs_model(y_valid, y_pred, class_name)

        #Interpretation
        fi_score = regre.feature_importance(X_valid, y_valid, selected_features)


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
