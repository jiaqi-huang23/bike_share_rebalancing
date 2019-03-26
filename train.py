# reference:
# 1. Model selection and evaluation - https://scikit-learn.org/stable/model_selection.html#model-selection
# 2. ML leanring map - https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

import joblib
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
import os
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

## TODO
# 1. Tuning hyper-parameters
# 2. implement models below:
#       SGD
#       Linear SVM
#       Naive Bayers
# 3. Model evaluation

def getValScore(clf,x,y):
    print("train-val")
    scores = cross_val_score(clf, x, y, cv=15, n_jobs=1, scoring = 'neg_median_absolute_error') 
    return (np.median(scores) * -1)

def train_status(config,x,y,test_x, test_y):
    # if config.model == "LogisticRegression":
    #     clf = LogisticRegression(
    #         random_state=0, 
    #         solver='lbfgs',
    #         multi_class='multinomial', 
    #         class_weight='balanced', 
    #         max_iter=200)
    # elif config.model == "SGDClassifier":
    #     clf = SGDClassifier(max_iter=1000, tol=1e-3)
    # elif config.model == "Lasso":
    #     clf = Lasso(alpha=0.1) 
    print("...start training status model....")

    # clf = svm.LinearSVC(class_weight='balanced')
    # clf = clf.fit(x, y)
    knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(x,y)
    # start training
    val_score = getValScore(knn,x, y)

    # score = accuracy_score(test_y,knn.predict(test_x))
    print(val_score)
    # return clf

    best_model = None
    # save best model
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    bestmodel_file = os.path.join(config.save_dir, "best_model.joblib")
    if os.path.exists(bestmodel_file):
        os.remove(bestmodel_file)
    else:
        joblib.dump(best_model, bestmodel_file)  


def train_difference(config,x,y,test_x, test_y):
    knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(x,y)
    # start training
    val_score = getValScore(knn,x, y)
    
    score = accuracy_score(test_y,knn.predict(test_x))
    print(score)
    # return clf
