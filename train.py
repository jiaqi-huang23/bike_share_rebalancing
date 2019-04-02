# reference:
# 1. Model selection and evaluation - https://scikit-learn.org/stable/model_selection.html#model-selection
# 2. ML leanring map - https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

import time
import csv
from sklearn.preprocessing import PolynomialFeatures
from collections import Counter
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
import os
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier, SGDRegressor,Perceptron
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, explained_variance_score,r2_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# TODO
# 1.random permute data before each validation

def get_models_names(multiclass_opt='multinomial'):
    models = [
       # GaussianNB(),
       # KNeighborsClassifier(),
       DecisionTreeClassifier(),
        #AdaBoostClassifier(),
        RandomForestClassifier(n_estimators=200),
        #LinearSVC(max_iter=1000),
        LogisticRegression(
            random_state=0,
            solver='lbfgs',
            multi_class=multiclass_opt,
            max_iter=1000
        ),
        SGDClassifier(max_iter=1000, tol=1e-3)
    ]

    names = [
           # "Naive Bayes",
           # "KNeighborsClassifier",
           "Decision Tree",
            # "AdaBoostClassifier",
            "RandomForestClassifier",
             # "LinearSVC",
             "LogisticRegression",
             "SGDClassifier"
            ]
    
    return models, names

def outputData(outfile,x,y,yPred, feature_names):
    with open(outfile, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(feature_names)
        for t in range(len(x)):
            tsv_writer.writerow(x[t]+y[t]+yPred[t].tolist())
            

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
    print('Original dataset shape %s' % Counter(y))
    print(x.shape)
    x, y = NearMiss().fit_resample(x, y)
    print('Resampled dataset shape %s' % Counter(y))
    print(x.shape)
    
    models, names = get_models_names()
   

    for model, name in zip(models, names):
        print(name)
        start = time.time()
        feature_names = ["station_id", "hour", "avg(temp)", "avg(wind)", "weekno"]
        m = np.mean(cross_val_score(model, x, y, scoring=my_scorer, cv=3))
        model = model.fit(x, y)

        if name == 'RandomForestClassifier' or name == 'LogisticRegression':
            yPred = model.predict(test_x)
            yPred = np.asarray(yPred)
            y2 = yPred[yPred==2]
            y1 = yPred[yPred==1]
            y3 = yPred[yPred==0]
            # for y_, predY_ in zip(test_y,yPred):
            #     print("truth lable: %d pred: %d" % (y_, predY_))
            #         print('=====')
            # print(y1)
            # print(y2)
            # print(y3)
            # print('=====')
            # print(test_x)
            # print(test_y.shape)
            # print(test_y)
            # print(yPred)
            # outputData('./rf_point_test.csv',test_x,test_y,yPred,feature_names)
            # importances = model.feature_importances_
            # print(importances)
            # # std = np.std([tree.feature_importances_ for tree in model.estimators_],
            # #             axis=0)
            # indices = np.argsort(importances)[::-1]

            # # Print the feature ranking
            # print("Feature ranking:")

            # for f in range(x.shape[1]):
            #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
            # feature_names = ["station_id", "hour", "avg(temp)", "avg(wind)", "weekno"]
            # fn = rearrange_feature(feature_names, indices)

            # # Plot the feature importances of the forest
            # plt.figure()
            # plt.title("Feature importances")
            # c = np.random.rand() 
            # plt.bar(range(x.shape[1]), importances[indices],
            #     color='#afeeee', align="center")
            # plt.xticks(range(x.shape[1]),  fn)
            # plt.xlim([-1, x.shape[1]])
            # plt.show()

        print('\nSum:', m, '\n\n')
        print("test")

        scores = my_scorer(model, test_x, test_y)
        print('time', time.time() - start, '\n\n')

    # clf.fit(x,y)
    # score = recall_score(test_y,clf.predict(test_x),average="weighted")
    # print(score)
    # return clf

    # best_model = None
    # # save best model
    # if not os.path.exists(config.save_dir):
    #     os.makedirs(config.save_dir)
    # bestmodel_file = os.path.join(config.save_dir, "best_model.joblib")
    # if os.path.exists(bestmodel_file):
    #     os.remove(bestmodel_file)
    # else:
    #     joblib.dump(best_model, bestmodel_file)



def getRegressionErrorScore(estimator, x, y):
    yPred = estimator.predict(x)
    # plt.scatter(np.mean(x,axis=1), y,  color='black')
    # plt.plot(x, yPred, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())
    # plt.show()
    return (mean_squared_error(y,yPred),
            mean_absolute_error(y,yPred),
            explained_variance_score(y,yPred),
            r2_score(y,yPred))


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

    models,names = get_models_names(multiclass_opt='ovr')
    for model, name in zip(models, names):
        print(name)
        start = time.time()

        m = np.mean(cross_val_score(model, x, y, scoring=my_scorer, cv=5))
        model = model.fit(x, y)


        print('\nSum:', m, '\n\n')
        print("test")
        scores = my_scorer(model, test_x, test_y)
        print('time', time.time() - start, '\n\n')

