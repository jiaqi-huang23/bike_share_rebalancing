# reference:
# 1. Model selection and evaluation - https://scikit-learn.org/stable/model_selection.html#model-selection
# 2. ML leanring map - https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

import time
import joblib
from collections import Counter
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict
import os
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


## TODO
# 1. Tuning hyper-parameters
# 2. Find out models that can fit:
#       SGD
#       Linear SVM
#       Naive Bayers
#       K nearest neighbours
# 3. Model evaluation

def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (accuracy_score(y, yPred), 
            precision_score(y, yPred, average='macro'), 
            recall_score(y, yPred, average='macro'))

def my_scorer(estimator, x, y):
    a, p, r = getScores(estimator, x, y)
    print (a, p, r)
    return a+p+r

def train_status(config,x,y,test_x, test_y):
    print("...start training status model....")
    print('Original dataset shape %s' % Counter(y))
    print(x.shape)
    x,y = NearMiss().fit_resample(x,y)
    print('Resampled dataset shape %s' % Counter(y))
    print(x.shape)

    models = [
                GaussianNB(), 
                KNeighborsClassifier(),
                DecisionTreeClassifier(),
                AdaBoostClassifier(),
                RandomForestClassifier(n_estimators=20),
                LinearSVC(),
                LogisticRegression(
                                    random_state=0, 
                                    solver='lbfgs',
                                    multi_class='multinomial', 
                                    max_iter=300
                                    ),
                SGDClassifier(max_iter=1000, tol=1e-3)
    ]
    names = ["Naive Bayes", 
            "KNeighborsClassifier", 
            "Decision Tree", 
            "AdaBoostClassifier", 
            "RandomForestClassifier",
            "LinearSVC",
            "LogisticRegression", 
            "SGDClassifier"]

    
    for model, name in zip(models, names):
        print (name)
        start = time.time()
        m = np.mean(cross_val_score(model, x, y,scoring=my_scorer, cv=15))
        model = model.fit(x,y)
        print ('\nSum:',m, '\n\n')
        print("test")
        scores = my_scorer(model, test_x, test_y)
        print ('time', time.time() - start, '\n\n')

    # clf.fit(x,y)
    # score = recall_score(test_y,clf.predict(test_x),average="weighted")
    # print(score)
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
    return
    # return clf
