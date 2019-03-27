# load, preprocess and wrap our data here
import pandas as pd
from sklearn import preprocessing

# TODO
# merge diff with weather on joining date


def loadStatusData(config):
    md = pd.read_csv("./data/final_day_merged.csv")

    # randomly sample x% of data to downsize dataset
    md = md.sample(frac=config.frac, replace=True)
    # take the fi
    # classification for station
    # split data into data and label
    labels = md.label
    data = md.drop(['label','year','month','day','date','fullness','holiday','weeknoref'],1)
    print(data.head(5))
    # convert data type to float (from string)
    data = data.astype(dtype=float)
    # scale data 
    data = preprocessing.scale(data)

    return data, labels

def loadDifferenceData(config):
    diff = pd.read_csv("./data/diff.csv")
    # weather = pd.read_csv("./data/weather.csv")

    # linear regression for diff
    result = diff.difference
    data = diff.drop(['date','year','month','day'],1)

    # convert data type to float (from string)
    data = data.astype(dtype=float)

    # scale data 
    data = preprocessing.scale(data)

    return data, result