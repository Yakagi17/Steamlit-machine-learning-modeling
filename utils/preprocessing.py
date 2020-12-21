from sklearn.preprocessing import LabelEncoder

def classEncoder(column, class_name):
    label_Encoder = LabelEncoder()
    label_Encoder.fit(class_name)
    colum = label_Encoder.transform(column)

    return colum