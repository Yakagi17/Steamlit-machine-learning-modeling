# from pyecharts import options as opts
# from pyecharts.charts import Bar

def feature_importance_plot(fi_score):
    #Sign feature name and permutation_score
    feature_name, permutation_score = [[name for name, _ in fi_score],[score for _, score in fi_score]]
    
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



'''
*note : still need to config legend auc_roc(variabel) score 
'''
def auc_plot(class_name, auc_score, auc_roc):    
    #Define dot slash line
    num_class = len(class_name)
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

    #Define auc line for every class
    for i in range(num_class):
        series_data_auc[i]= {
            'name': class_name[i],
            'data': auc_score[i],
            'type': 'line',
            'areaStyle': {}
        }
    
    for i in range(num_class):
        series_options.append(series_data_auc[i])

    #Set ROC AUC Curve Format
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

def coordinates_plot():
    pass
    
