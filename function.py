# from pyecharts import options as opts
# from pyecharts.charts import Bar

def feature_importance_plot(feature_name, permutation_score):
    #Sort Permutation importance scoe on descending order
    zip_fi = zip(feature_name, permutation_score)
    sorted_zip_fi = sorted(zip_fi, key=lambda x:x[1], reverse=False)
    feature_name, permutation_score = [[name for name, _ in sorted_zip_fi],[score for _, score in sorted_zip_fi]]
    
    options = {
        "title": {
            "text": 'Feature Importance'
            # subtext: 'feature importance of ... dataset using permuation importance'
            },
        "tooltip": {
            "trigger": 'axis',
            "axisPointer": {
                "type": 'shadow'
                }
            },
        "grid": {
            "left": '3%',
            "right": '4%',
            "bottom": '3%',
            "containLabel": "true"
            },
        "xAxis": {
            "type": 'value',
            "boundaryGap": [0, 0.01]
            },
        "yAxis": {
            "type": 'category',
            "data": feature_name
            },
        "series": [
        {
            "name": 'Permutaion Importance Score',
            "type": 'bar',
            "data": permutation_score
        }
        ]
    }

    return options

from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder

def auc_plot(class_name, y_true, y_proba):
    fpr = {}
    tpr = {}
    thresh = {}
    auc_score = []
    num_class = len(class_name)

    #Check if y value is String or Object type
    if y_true.dtype == 'O':
        label_Encoder = LabelEncoder()
        label_Encoder.fit(class_name)
        y_true = label_Encoder.transform(y_true)
    #Calcluate true positif rate and false positif rate
    for i in range(num_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_true, y_proba[:,i], pos_label=i)
    
    for i in range(num_class):
        temp = []
        for j in range(len(fpr[i])):
            temp.append([fpr[i][j], tpr[i][j]])
        auc_score.append(temp)
    
    #Plot AUC ROC Curve
    series_data_auc = {}
    series_options = [{
            'data': [0,1],
            'type': 'line',
            'lineStyle': {
                'color': '#5470C6',
                'width': 2,
                'type': 'dashed'
                }
            }]

    for i in range(num_class):
        series_data_auc[i]= {
            'name': class_name[i],
            'data': auc_score[i],
            'type': 'line',
            'areaStyle': {}
        }
    
    for i in range(num_class):
        series_options.append(series_data_auc[i])

    options = {
        'title': {
            'text': 'Area Under ROC Curve'
            },
        'legend': {
            'top': 'top',
            'data': class_name,
            'top':'bottom'
            },
        'xAxis': {
            'name': 'False Positive Rate',
            'nameLocation': 'middle',
            'nameTextStyle': {
                'fontWeight': "bold",
                'verticalAlign': "top",
                'lineHeight': 30
                }
            },
        'yAxis': {
            'name': 'True Positive Rate',
            'nameLocation': 'middle',
            'nameTextStyle': {
                'fontWeight': "bold",
                'verticalAlign': "bottom",
                'lineHeight': 50
                }
            },
        'series': series_options
        }

    return options



def partial_dependence_plot(axes, pdp):
    pass

    
