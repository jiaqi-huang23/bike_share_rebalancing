# reference:
# 1. Model selection and evaluation - https://scikit-learn.org/stable/model_selection.html#model-selection
# 2. ML leanring map - https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

import joblib
from sklearn.model_selection import train_test_split
import os

## TODO
# 1. Tuning hyper-parameters
# 2. implement models below:
#       SGD
#       Linear SVM
#       Naive Bayers
# 3. Model evaluation




# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.10, random_state=2)

# start training


# save best model
    bestmodel_file = os.path.join(config.save_dir, "best_model.joblib")
    if os.path.exists(bestmodel_file):
        os.remove(bestmodel_file)
    else:
        joblib.dump(to_persist, bestmodel_file)  


