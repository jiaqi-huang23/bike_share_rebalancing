# reference:
# 1. Model selection and evaluation - https://scikit-learn.org/stable/model_selection.html#model-selection
# 2. ML leanring map - https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

import time
from sklearn.preprocessing import PolynomialFeatures
from collections import Counter
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
import os
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


def get_models_names(multiclass_opt='multinomial'):
    models = [
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=200),
        LogisticRegression(
            solver='lbfgs',
            multi_class=multiclass_opt,
            max_iter=3000
        ),
        SGDClassifier(max_iter=10000, alpha=0.00001, tol=1e-3)
    ]

    names = [
        "Decision Tree",
        "RandomForestClassifier",
        "LogisticRegression",
        "SGDClassifier"
    ]

    return models, names


def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (accuracy_score(y, yPred),
            precision_score(y, yPred, average='macro'),
            recall_score(y, yPred, average='macro'))


def my_scorer(estimator, x, y):
    a, p, r = getScores(estimator, x, y)
    print(a, p, r)
    return a+p+r


def train_status(config, x, y, test_x, test_y):
    print("...start training status model....")
    # balance data
    print('Original dataset shape %s' % Counter(y))
    print(x.shape)
    x, y = NearMiss().fit_resample(x, y)
    print('Resampled dataset shape %s' % Counter(y))
    print(x.shape)
    models, names = get_models_names()

    for model, name in zip(models, names):
        print(name)
        start = time.time()
        feature_names = ["station_id", "hour",
                         "avg(temp)", "avg(wind)", "weekno"]

        if name == 'LogisticRegression':
            model = model.fit(x, y)
            train_score = model.score(x, y)
            print(train_score)
            print("test")
            score = model.score(test_x, test_y)
            accuracy = accuracy_score(test_y, model.predict(test_x))
            print(score)
            print(accuracy)
            print('time', time.time() - start, '\n\n')
        else:
            m = cross_validate(model, x, y, scoring=my_scorer,
                               cv=10, return_estimator=True)
            model = m['estimator'][-1]
            model = model.fit(x, y)

            if name != 'SGDClassifier':
                # plot weight importance
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]

                # Print the feature ranking
                print("Feature ranking:")

                for f in range(x.shape[1]):
                    print("%d. feature %d (%f)" %
                          (f + 1, indices[f], importances[indices[f]]))
                feature_names = ["station_id", "hour",
                                 "month", "avg(temp)", "avg(wind)", "weekno"]
                fn = rearrange_feature(feature_names, indices)

                # Plot the feature importances of the forest
                plt.figure()
                plt.title("Feature importances")
                plt.bar(range(x.shape[1]), importances[indices],
                        color='#4682B4', align="center")
                plt.xticks(range(x.shape[1]),  fn)
                plt.xlim([-1, x.shape[1]])
                plt.show()

            print("test")
            scores = my_scorer(model, test_x, test_y)
            print(scores)
            print('time', time.time() - start, '\n\n')


def rearrange_feature(feature_names, index):
    f = []
    for i in index:
        f.append(feature_names[i])
    return f


def train_difference(config, x, y, test_x, test_y):
    print("...start training difference model....")
    print('Original dataset shape %s' % Counter(y))
    print(x.shape)
    x, y = NearMiss().fit_resample(x, y)
    print('Resampled dataset shape %s' % Counter(y))
    print(x.shape)

    models, names = get_models_names(multiclass_opt='ovr')
    for model, name in zip(models, names):
        print(name)
        start = time.time()

        if name == 'LogisticRegression':
            model = model.fit(x, y)
            print("test")
            score = model.score(test_x, test_y)
            print(score)
            print('time', time.time() - start, '\n\n')
        else:
            m = cross_validate(model, x, y, scoring=my_scorer,
                               cv=10, return_estimator=True)
            model = m['estimator'][-1]
            model = model.fit(x, y)

            print("test")
            scores = my_scorer(model, test_x, test_y)
            print(scores)
            print('time', time.time() - start, '\n\n')
