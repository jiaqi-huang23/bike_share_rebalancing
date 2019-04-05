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

def getTargetSample(df,station_key):
    stat_id = ['32', '66', '62', '82'] 
    test_points = df.loc[
        (df[station_key].isin(stat_id)) & 
        (df['year'].isin(['2015'])) & 
        (df['month'].isin(['7'])) &
        (df['day'].isin(['23'])) &
        (df['hour'].isin(['10']))
    ]

    return test_points

def loadStatusData(config):
    md = pd.read_csv("./data/final_data.csv")
    # md['weekno'] = md.apply(lambda row: merge_day(row), axis=1)
    # md.to_csv(index=False,path_or_buf='./data/final_data_2.csv')
    # randomly sample x% of data to downsize dataset
    # md = md.sample(frac=config.frac, replace=True)

    sampled_points = getTargetSample(md,'station_id')
    # print(sampled_points)
    # take the fi
    # classification for station
    # split data into data and label
    labels = md.label
    data = md.drop(['label','avg(bikes_available)','year', 'avg(docks_available)','month','day','date','fullness','holiday'],1)
    print(data.head(5))
    # convert data type to float (from string)
    data = data.astype(dtype=float)
    labels = labels.astype(dtype=int)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    # print(mean)
    # print(std)
    # scale data
    data = preprocessing.scale(data)
    sample_lables = (sampled_points.label).astype(dtype=int)
    # print(sample_lables)
    sample_data = sampled_points.drop(['label','avg(bikes_available)',
                'avg(docks_available)', 'max(rain)','year', 'month','day','date','fullness','holiday'],1).astype(dtype=float)
    sample_data = sample_data.apply(lambda row: scale(row,mean,std), axis=1)
    # print(arr)
    # arr = (arr-mean)/std
    # print(arr)
    # # sample_data = preprocessing.normalize(sample_data)
    # # print(sample_data)

    return data, labels, sample_data, sample_lables

def labelInOut(row):
    if row['difference'] >= 0:
        return 1
    else:
        return 0

def loadDifferenceData(config):
    diff = pd.read_csv("./data/diff_weather.csv")
    # diff = diff.sample(frac=config.frac, replace=True)
    # diff['difference'] = diff.apply(lambda row: labelInOut(row), axis=1)
    # diff.to_csv(index=False,path_or_buf='./data/diff_weather2.csv')
    # randomly sample x% of data to downsize dataset
    # linear regression for diff
    
    sampled_points = getTargetSample(diff,'station')
    #print(sampled_points)
    label = diff.difference
    data = diff.drop(['date','year','month','day','difference','weeknoRef','holiday'],1)
    
    sample_lables = sampled_points.difference
    sample_data = sampled_points.drop(['date','year','month','day','difference','weeknoRef','holiday'],1)
    print(data.head())

    # convert data type to float (from string)
    data = data.astype(dtype=float)
    label = label.astype(dtype=int)
    # scale data
    data = preprocessing.scale(data)
    
    return data, label, sample_data, sample_lables
