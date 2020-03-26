### IMPORT LIBRARIES ###
import numpy as np
import pandas as pd
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

### IMPORT DATASETS ###
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
y_test = pd.read_csv('gender_submission.csv')

#### SPLIT DATASET #####
# Creating working training data
ds_train = dataset_train
y_train = dataset_train.iloc[:, 1].values
# Creating working test data
ds_test = dataset_test
y_id = y_test.iloc[:,0]
y_id = pd.Series(y_id).to_numpy()
y_test = y_test.iloc[0:, 1]
#Creating pack for data cleaning
X_pack = [ds_train,ds_test ]
# Filling missing data
for dataset in X_pack:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

ds_train.info()

#Delete unwanted columns
drop_column = ['PassengerId','Cabin', 'Ticket']
ds_train.drop(drop_column, axis=1, inplace = True)
ds_test.drop(drop_column, axis=1, inplace = True)

for dataset in X_pack:    
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels= ['cheap', 'medium', 'high', 'expensive'])
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5, labels= ['kid', 'young_adult', 'adult', 'mature', 'old'])
#cleanup rare title names
stat_min = 10 
title_names = (ds_train['Title'].value_counts() < stat_min) # True/False separation

ds_train['Title'] = ds_train['Title'].apply(lambda x: 'Unique' if title_names.loc[x] == True else x)
print(ds_train['Title'].value_counts())

#define y variable aka target/outcome
Target = ['Survived']
#Define list of categorical variable columns
cat_var = ['Sex', 'Embarked', 'Title', 'FareBin', 'AgeBin']

#Delete unwanted columns
drop_column_2 = ['Age','SibSp', 'Parch', 'Fare', ]
ds_train.drop(drop_column_2, axis=1, inplace = True)
ds_test.drop(drop_column_2, axis=1, inplace = True)   
#Create dummy varieables
ds_train_dummy = pd.get_dummies(ds_train[cat_var], drop_first=True)
ds_test_dummy = pd.get_dummies(ds_train[cat_var], drop_first=True)
# Combine dummy and standard dataset
ds_train = pd.concat([ds_train, ds_train_dummy], axis=1, sort=False)
ds_test = pd.concat([ds_test, ds_test_dummy], axis=1, sort=False)
#Delete unwanted columns
drop_column_3 = ['Sex', 'Embarked', 'Title', 'FareBin', 'AgeBin', 'Name']
ds_train.drop(drop_column_3, axis=1, inplace = True)
ds_test.drop(drop_column_3, axis=1, inplace = True)
#List of train columns
train_columns = ['Pclass', 'FamilySize', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Title_Miss', 'Title_Mr','Title_Mrs', 'Title_Unique',
                 'FareBin_medium', 'FareBin_high', 'FareBin_expensive', 'AgeBin_young_adult', 'AgeBin_adult', 'AgeBin_mature'
                 ,'AgeBin_old']
#split train and test data with function defaults
X_train, X_test, y_train, y_test = model_selection.train_test_split(ds_train[train_columns], ds_train[Target], random_state = 0)

#### MODEL DATA
#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]
#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = ds_train[Target]

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, ds_train[train_columns], ds_train[Target], cv  = cv_split, return_train_score=True)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions - see section 6 for usage
    alg.fit(ds_train[train_columns], ds_train[Target])
    MLA_predict[MLA_name] = alg.predict(ds_train[train_columns])
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare

# Creating a model





