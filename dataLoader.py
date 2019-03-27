# load, preprocess and wrap our data here
import pandas as pd
from sklearn import preprocessing

# TODO
# merge diff with weather on joining date

def loadStatusData():
    md = pd.read_csv("./data/final_data.csv")
    
    # classification for station
    # split data into data and label
    labels = md.label
    data = md.drop(['label','year','month','day','date'],1)

    # convert data type to float (from string)
    data = data.astype(dtype=float)

    # scale data 
    data = preprocessing.scale(data)

    return data, labels

def loadDifferenceData():
    diff = pd.read_csv("./data/diff_weather.csv")

    print(diff.head())
    # linear regression for diff
    result = diff.difference
    data = diff.drop(['date','year','month','day'],1)

    # convert data type to float (from string)
    data = data.astype(dtype=float)

    # scale data 
    data = preprocessing.scale(data)

    return data, result