'''
Importing some important library
'''
#Mian web App
import streamlit as st

#Web Chart
from streamlit_echarts import st_echarts, st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Bar

#Machine Learning 
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor

# from sklearn.cluster import KMeans


#Dataset
data_url = "dataset/iris.csv"
classification_df = pd.read_csv(data_url)
# regression_df = pd.read_csv("dataset/california_housing_train.csv")
# clustering_df = pd.read_csv("dataset/Mall_Customers.csv")

# st.dataframe(classification_df)
# st.dataframe(regression_df)
# st.dataframe(clustering_df)


#Preprocess dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# SPLIT_RATIO = 0.8
processed_features = ['sepal length', 'sepal width', 'petal length', 'petal width']
target = ['species']

X = classification_df[processed_features].values
y = classification_df[target].values
class_name = classification_df.species.unique().tolist()
feature_names = classification_df.columns[:-1].tolist()
# X_train, X_valid, y_train, y_valid= train_test_split(X, y, test_size=SPLIT_RATIO, shuffle=False)



#Modeling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier

clf_model = SVC(probability=True, random_state = 101)
clf_model.fit(X, y)

#Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

result = clf_model.predict(X)
result_proba = clf_model.predict_proba(X)
accuracy = round(accuracy_score(y, result), 2)
conf_matrix = confusion_matrix(y, result) #tolist()
clf_report = classification_report(y, result, output_dict=True)



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

st.title('Machine Learning Modeling - Demo')
st.text('Demo classification modeling with SVM(Support Vector Machine) using Iris dataset')

st.subheader('Iris Dataset')

st.dataframe(classification_df)

st.header('Evaluation :')
st.text(f'Classification accuracy of this SVM model is : {accuracy*100} %')
st.table(conf_matrix)
st.table(pd.DataFrame(clf_report))

st.header('Interpretation :')

'''
from utils.plotting import feature_importance_plot

pi_plot =  feature_importance_plot(fi_score)

st_echarts(pi_plot)

from utils.plotting import auc_plot
auc_roc_plot = auc_plot(class_name, y, result_proba)

st_echarts(auc_roc_plot)
'''































# options = {
#     "xAxis":{
#         "type": "category",
#         "data": ["Mon", "Tue", "Wed"],
#     },
#     "yAxis" :{
#         "type": "value"
#     },
#     "series": [
#         {
#             "data": [100, 120, 140],
#             "type": "line"
#         }
#     ]
# }

# st_echarts(options=options)

# from pyecharts import options as opts
# from pyecharts.charts import Bar
# from streamlit_echarts import st_pyecharts

# b = (
#     Bar()
#     .add_xaxis(["Microsoft", "Amazon", "IBM", "Oracle", "Google", "Alibaba"])
#     .add_yaxis(
#         "2017-2018 Revenue in (billion $)", [21.2, 20.4, 10.3, 6.08, 4, 2.2]
#     )
#     .set_global_opts(
#         title_opts=opts.TitleOpts(
#             title="Top cloud providers 2018", subtitle="2017-2018 Revenue"
#         ),
#         toolbox_opts=opts.ToolboxOpts(),
#     )
# )
# st_pyecharts(b)