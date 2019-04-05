# load, preprocess and wrap our data here
import pandas as pd
from sklearn import preprocessing
import numpy as np

# TODO
# merge diff with weather on joining date


def merge_day(row):
    if row['holiday'] == 1:
        return 0
    else:
        return row['weekno']


def scale(row, mean, std):
    return (row - mean)/std


def loadStatusData(config):
    md = pd.read_csv("./data/final_data.csv")
    # randomly sample x% of data to downsize dataset
    # md = md.sample(frac=config.frac, replace=True)

    # split data into data and label
    labels = md.label
    data = md.drop(['label', 'avg(bikes_available)', 'year', 'avg(docks_available)','max(rain)',
                     'day', 'date', 'fullness', 'holiday'], 1)
    print(data.head(5))
    # convert data type to float (from string)
    data = data.astype(dtype=float)
    labels = labels.astype(dtype=int)
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    # scale data
    data = preprocessing.scale(data)

    return data, labels


def labelInOut(row):
    if row['difference'] >= 0:
        return 1
    else:
        return 0


def loadDifferenceData(config):
    diff = pd.read_csv("./data/diff_weather.csv")
    # diff = diff.sample(frac=config.frac, replace=True)
    # diff['difference'] = diff.apply(lambda row: labelInOut(row), axis=1)
    # randomly sample x% of data to downsize dataset
    # linear regression for diff

    #print(sampled_points)
    label = diff.difference
    data = diff.drop(['date', 'year', 'day',
                      'difference', 'weeknoRef', 'holiday'], 1)

    print(data.head())

    # convert data type to float (from string)
    data = data.astype(dtype=float)
    label = label.astype(dtype=int)
    # scale data
    data = preprocessing.scale(data)

    return data, label
