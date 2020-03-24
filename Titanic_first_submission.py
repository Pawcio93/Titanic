# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
y_test = pd.read_csv('gender_submission.csv')
y_id = y_test.iloc[:,0]
y_id = pd.Series(y_id).to_numpy()
y_test = y_test.iloc[0:, 1]

dataset_train = dataset_train.fillna(dataset_train.mean())
dataset_test = dataset_test.fillna(dataset_test.mean())



#### PREPROCESSING #####

# Creating working training data
X_train = dataset_train.iloc[:, [2, 4, 5, 6, 7, 9]].values
y_train = dataset_train.iloc[:, 1].values
X_test = dataset_test.iloc[:, [1, 3, 4, 5, 6, 8]].values

### Feature scalling
# creating temp sets
scaled_feat_train = X_train[:, [2,5]]
scaled_feat_test = X_test[:, [2,5]]
# Applying standard scalling
from sklearn.preprocessing import StandardScaler
sc_feat = StandardScaler()
scaled_feat_train = sc_feat.fit_transform(scaled_feat_train)
scaled_feat_test = sc_feat.transform(scaled_feat_test)
# Applying modificated data to original datasets
X_train[:, [2,5]] = scaled_feat_train
X_test[:, [2,5]] = scaled_feat_test

### Encoding data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_work_train_data = LabelEncoder()
labelencoder_test_data = LabelEncoder()
## fit and transform labels
X_train[:,1] = labelencoder_work_train_data.fit_transform(X_train[:,1])
X_test[:,1] = labelencoder_test_data.fit_transform(X_test[:,1])

# set encoding parameters for OneHotEncoder
ct_train_sex = ColumnTransformer([("Sex", OneHotEncoder(), [1])],    remainder = 'passthrough')
ct_test_sex = ColumnTransformer([("Sex", OneHotEncoder(), [1])],    remainder = 'passthrough')
X_train = ct_train_sex.fit_transform(X_train)
X_test = ct_test_sex.fit_transform(X_test)
# deleting one dummy column
X_train = np.delete(X_train, 0, 1)
X_test = np.delete(X_test, 0, 1)

# set encoding parameters for OneHotEncoder
ct_train_class = ColumnTransformer([("Pclass", OneHotEncoder(), [1])],    remainder = 'passthrough')
ct_test_class = ColumnTransformer([("Pclass", OneHotEncoder(), [1])],    remainder = 'passthrough')
X_train = ct_train_class.fit_transform(X_train)
X_test = ct_test_class.fit_transform(X_test)
# deleting one dummy column
X_train = np.delete(X_train, 0, 1)
X_test = np.delete(X_test, 0, 1)

# Fitting XGBoost to the training 
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth = 4, learning_rate = 0.1, n_estimators=135, booster= 'gbtree', gamma = 0.1)
classifier.fit(X_train, y_train)

#predicting the test set results
y_pred = classifier.predict(X_test)
y_pred_id = np.column_stack((y_id,y_pred))
y_pred_id = pd.DataFrame({'PassengerId': y_pred_id[:, 0], 'Survived': y_pred_id[:, 1]})

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# # Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv = 10)
accuracies.mean()
accuracies.std()

# # Applying Grid search - to find best model and best params
from sklearn.model_selection import GridSearchCV
parameters = [{'max_depth':[4, 5], 'learning_rate':[0.05, 0.1, 0.15], 'n_estimators':[115, 120, 125, 130, 135],
               'gamma':[0.05, 0.1, 0.15],'booster':['gbtree']},]
grid_search = GridSearchCV(estimator=classifier,
                             param_grid=parameters,
                             scoring = 'accuracy',
                             cv = 10)
grid_search.fit(X_train, y_train)
best_acc = grid_search.best_score_
best_params = grid_search.best_params_

y_pred_id.to_csv('y_pred.csv', index = False)
