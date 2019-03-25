import joblib
from sklearn.model_selection import train_test_split
import os
from dataLoader import loadStatusData, loadDifferenceData
from train import train_status, train_difference
import config

def main():
    # load data here, then train and test
    status_data, status_label = loadStatusData()
    diff_data, diff_res = loadDifferenceData()

    # split data into train and test
    status_train, status_test, status_lable_train, status_label_test = train_test_split(
        status_data,
        status_label,
        test_size=0.10,
        random_state=2)

    diff_train, diff_test, diff_res_train, diff_res_test = train_test_split(
        diff_data,
        status_label,
        test_size=0.10,
        random_state=2)

    # train
    train_status(config,status_train, status_lable_train)
    train_difference(config, diff_train, diff_res_train)

    # test


if __name__ == "__main__":
    main()
