import joblib
from sklearn.model_selection import train_test_split
import os
from dataLoader import loadStatusData, loadDifferenceData
from train import train_status, train_difference
from config import get_config, print_usage

def main(config):
    # load data here, then train and test
    status_data, status_label = loadStatusData()
    diff_data, diff_res = loadDifferenceData()

    print(status_data.shape)
    print(status_label.shape)
    # split data into train and test
    status_train, status_test, status_lable_train, status_label_test = train_test_split(
        status_data,
        status_label,
        test_size=0.10,
        random_state=2)

    diff_train, diff_test, diff_res_train, diff_res_test = train_test_split(
        diff_data,
        diff_res,
        test_size=0.10,
        random_state=2)

    # train
    train_status(config,status_train, status_lable_train, status_test, status_label_test)

    # train_difference(config, diff_train, diff_res_train, diff_test, diff_res_test)

    # test

if __name__ == "__main__":
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
