import pandas as pd
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score
import statistics
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
from time import time
import joblib
import json


import warnings
warnings.filterwarnings("ignore")

def cls_alg(CLS_ALGORITHM,
            train_df, dev_df,
            columns, target,feature_extraction=True,path_to_save='../experiments/',
            report=True, visualization=True):
    
    model, y_test, predicted, best_params, pipeline = text_cls(CLS_ALGORITHM,
                                 train_df[columns], train_df[target],
                                 dev_df[columns], dev_df[target],
                                 feature_extraction=True,path_to_save_model=path_to_save)
           
    metrics_dict = count_metrics(y_test, predicted)
    
    print(metrics_dict)
    
    if report:
            print(classification_report(y_test, predicted))
            print('\n  Visualization of classification')
    if visualization:
        visualize(y_test, predicted, dev_df, target, CLS_ALGORITHM, image_path='../experiments')
        
    return metrics_dict, model, y_test, predicted, pipeline



def text_cls(CLS_ALGORITHM,
             X_train, y_train,
             X_test, y_test,
             feature_extraction=True,
             path_to_save_model='../experiments'):
    print(f"-----{CLS_ALGORITHM}-----")
    
    if CLS_ALGORITHM == 'LogisticRegression':
        nb = Pipeline([('clf', LogisticRegression())])
        parameters_nb = {'clf__penalty': ( "l1", "l2", "elasticnet"),
                         'clf__C': (1.5, 1, 0.5),
                         'clf__class_weight': ['balanced'],
                        'clf__solver': ['lbfgs', 'liblinear', 'newton-cholesky', 'saga']}
        
    elif CLS_ALGORITHM == 'MultinomialNB':
        nb = Pipeline([('clf',MultinomialNB())])
        parameters_nb = {'clf__alpha': ( 0.01, 0.005, 0.001)}
    
    elif CLS_ALGORITHM == 'PassiveAggressiveClassifier':
        nb = Pipeline([('clf', PassiveAggressiveClassifier())])
        parameters_nb = {'clf__max_iter': (100, 500, 1000, 1500),
                        'clf__class_weight': ['balanced', None],
                        'clf__loss': ['hinge', 'squared_hinge']}
        
    elif CLS_ALGORITHM =='RandomForestClassifier':
        nb = Pipeline([('clf', RandomForestClassifier())])
        parameters_nb = {'clf__criterion': ["gini", "entropy", "log_loss"],
                        'clf__n_estimators': [10, 50, 100, 150, 200],
                        'clf__class_weight': ['balanced', 'balanced_subsample'],
                        'clf__ccp_alpha': [0.0, 0.05, 0.001]}
        
    elif CLS_ALGORITHM =='DecisionTreeClassifier':
        nb = Pipeline([('clf', DecisionTreeClassifier())])
        parameters_nb = {'clf__criterion': ["gini", "entropy", "log_loss"],
                        'clf__splitter': ["best", "random"],
                        'clf__min_impurity_decrease': [0.0, 0.01, 0.1],
                        'clf__class_weight': ['balanced', None],
                        'clf__ccp_alpha': [0.0, 0.05, 0.001]}
        
    elif CLS_ALGORITM == 'GradientBoostingClassifier':
        nb = Pipeline([('clf', GradientBoostingClassifier())])
        parameters_nb = {'clf__loss': ["log_loss", "exponential"],
                        'clf__laerning_rate': [0.1, 0.05, 0.001],
                        'clf__n_estimators': [100, 50, 200],
                        'clf__criterion': ['friedman_mse', 'squared_error']}
        
        
    gs_clf_nb = GridSearchCV(nb, parameters_nb, n_jobs=-1, scoring = "f1_weighted")
    gs_clf_nb = gs_clf_nb.fit(X_train, y_train)
    print('best score --- ', gs_clf_nb.best_score_)
    print('best parameters --- ', gs_clf_nb.best_params_)
    r = gs_clf_nb.best_estimator_
    predicted = r.predict(X_test)
    print('-----------------------------------------')
    joblib.dump(r, f'{path_to_save_model}/models/{CLS_ALGORITHM}.pkl')
    if feature_extraction:
        return r, y_test, predicted, gs_clf_nb.best_params_, gs_clf_nb
    
def visualize(y_true, y_pred, df, target, algorithm, image_path='../experiments'):
    plt.figure(figsize=(14,10))
    array=confusion_matrix(y_true, y_pred)
    a = [sorted(df[target].unique())]
    df_cm = pd.DataFrame(array, index=a, columns=a)
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={'size':9}, cmap='Blues',fmt='g')
    plt.xticks(rotation=60)
    plt.xlabel('Predictions')
    plt.ylabel('Real answers')
    plt.savefig(f'{image_path}/images/{algorithm}.png')
    plt.show()

    
def count_metrics(y_test, predicted, CLS_ALGORITHM):
    metrics_dict = {'algorithm': CLS_ALGORITHM,
                    'precision': round(precision_score(y_test, predicted), 2),
                    'recall': round(recall_score(y_test, predicted),2),
                    'f1_score': round(f1_score(y_test, predicted),2),
                    'f1_score_weighted': round(f1_score(y_test, predicted, average='weighted'),2),
                    'accuracy': round(accuracy_score(y_test, predicted), 2)}
    return metrics_dict
