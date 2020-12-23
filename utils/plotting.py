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
            # 'top':'bottom'
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

def coordinates_plot(centroid_df, selected_features, num_cluster):
    data_per_cluster = {}
    series_data = []
    parallel_axis_name = []

    for i in range(num_cluster):
        data_per_cluster[i] = centroid_df[centroid_df['cluster'] == i].values.tolist()
        temp_series_format = {"name": f"Cluster - {i}", "type": "parallel", "lineStyle": "lineStyle", "data": data_per_cluster[i]}
        series_data.append(temp_series_format)
    
    for i in range(len(selected_features)):
        temp_parallel_axis_format = {"dim": i, "name": selected_features[i]}
        parallel_axis_name.append(temp_parallel_axis_format)
    
    legend = {
        "top": 30,
        "data": selected_features,
        "itemGap": 20,
        "textStyle": {
            "color": '#fff',
            "fontSize": 14
            }
        }

    options = {
        "legend" : legend,
        "parallelAxis": parallel_axis_name,
        "series": series_data
    }

    return options
    
def cluster_data_coordinates_plot():
    pass
