#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 16:30:00 2025

@author: Kalle Lahtinen, kalle.t.lahtinen@tuni.fi

This script implements the functions to execute training and testing for a Support Vector Machine classifier. 
The expected data is read in from csv-files, that contain the training and testing features as well as the 
sample labels for high and low arousal as well as high, low and neutral valence. The script trains two separate 
classifiers for predicting the affect related discrete labels for arousal and valence.
    
"""

import numpy as np
import sys
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import classification_scores
from sklearn.metrics import make_scorer

#wrapping function that returns the UAR for the classifier test to be used in 
#the SVM parameter grid search  
def uar_for_scorer(y_true, y_pred):
    
    score = classification_scores.get_unweighed_average_recall(y_true, y_pred)
    
    return score["uar"]
    
if __name__ == '__main__':
    
    ###########################################################################
    
    #define the root dir for the data
    my_cwd = os.getcwd()
    data_storage_dir = my_cwd+"/data/"
    #continuous or discrete labels, in this script always discrete
    label_type = "discrete"
    #perform grid search or just train one classifier without grid search
    grid_search = False
        
    #Read in annotations for training (tr_val -files) and testing (gold_std -files) the classifier
    valence_gold_std = pd.read_csv(data_storage_dir+"/"+label_type+"_valence_gold_std.csv")
    valence_gold_std = valence_gold_std.set_index("Unnamed: 0")
    valence_tr_val = pd.read_csv(data_storage_dir+"/"+label_type+"_valence_tr_val.csv")
    valence_tr_val = valence_tr_val.set_index("Unnamed: 0")
    arousal_gold_std = pd.read_csv(data_storage_dir+"/"+label_type+"_arousal_gold_std.csv")
    arousal_gold_std = arousal_gold_std.set_index("Unnamed: 0")
    arousal_tr_val = pd.read_csv(data_storage_dir+"/"+label_type+"_arousal_tr_val.csv")
    arousal_tr_val = arousal_tr_val.set_index("Unnamed: 0")
    
    #training and testin features
    tr_val_features = pd.read_csv(data_storage_dir+"/"+"features_tr_val.csv")
    tr_val_features = tr_val_features.set_index("Unnamed: 0")
    GS_features = pd.read_csv(data_storage_dir+"/"+"features_gold_std.csv")
    GS_features = GS_features.set_index("Unnamed: 0")
    
    #sanity check for arousal and valence dataframes, should have the same 
    #sample ids
    if arousal_gold_std.index.to_list() != valence_gold_std.index.to_list():
        print("arousal and valence gold std different!")
        sys.exit()
        
    if arousal_tr_val.index.to_list() != valence_tr_val.index.to_list():
        print("arousal and valence training sets different!")
        sys.exit()
        
    if len(set(arousal_gold_std.index.to_list()).intersection(set(arousal_tr_val.index.to_list()))) > 0:
        print("Overlapping sample ids in arousal gld std and tr sets")
        sys.exit()
        
    if len(set(valence_gold_std.index.to_list()).intersection(set(valence_tr_val.index.to_list()))) > 0:
        print("Overlapping sample ids in valence gld std and tr sets")
        sys.exit()
        
    
    #valence training and testing labels
    valence_annotated_tr_labels = valence_tr_val.loc[~valence_tr_val["annotation_mode"].isna(), "a_propagated"]
    valence_GS_labels = valence_gold_std["annotation_mode"]
    
    #arousal training and testing labels
    arousal_annotated_tr_labels = arousal_tr_val.loc[~arousal_tr_val["annotation_mode"].isna(), "a_propagated"]
    arousal_GS_labels = arousal_gold_std["annotation_mode"]

    #Cast the data to numpy arrays, z-score normalize (zero mean, unit variance) the data
    X_train = tr_val_features.to_numpy()
    X_test = GS_features.to_numpy()
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    
    # Wrap the custom scoring function
    uar_scorer = make_scorer(uar_for_scorer)
    
    #SVM training and testing for arousal, 
    #get arousal labels
    y_train = arousal_annotated_tr_labels.to_numpy()
    y_test = arousal_GS_labels.to_numpy()
    
    #grid search for SVM parameters, 5 fold cross validation
    if grid_search: 
        clf = svm.SVC()
        
        svm_linear = {'C': [0.1, 1, 10], 
              'kernel': ['linear'],
              'class_weight': ["balanced"]} 
        svm_others = {'C': [0.1, 1, 10],
              'gamma': [1, 0.1, 0.01,'auto'], 
              'kernel': ['poly', 'rbf', 'sigmoid'],
              'class_weight': ["balanced"]}
        
        param_grid = [svm_linear, svm_others]
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=uar_scorer, verbose=1)
        
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)
        
        # Display best parameters and best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Validation Accuracy:", grid_search.best_score_)
        
        # Evaluate the best model on the test set
        best_model_ar = grid_search.best_estimator_
        best_model_ar.fit(X_train, y_train)
        y_pred_ar = best_model_ar.predict(X_test)
    
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred_ar))
     
    #train and test SVM classifier with constant parameters
    else:
        print("Fitting SVM for arousal with selected parameters")

        clf = svm.SVC(kernel="linear", C=1, class_weight="balanced")
        clf.fit(X_train, y_train)
        
        y_pred_ar = clf.predict(X_test)
    
    #compute confusion matrix for arousal
    ar_c_mat = confusion_matrix(y_test, y_pred_ar)
    #compute classification metrix for arousal
    arousal_metrics = classification_scores.get_all_scores(y_test, y_pred_ar)
    
    print(arousal_metrics)
    
    
    
    #SVM training and testing for valence
    y_train = valence_annotated_tr_labels.to_numpy()
    y_test = valence_GS_labels.to_numpy()
    
    #grid search for SVM parameters, 5 fold cross validation
    if grid_search: 
        clf = svm.SVC()
        # Define the parameter grid
        svm_linear = {'C': [0.1, 1, 10], 
              'kernel': ['linear'],
              'class_weight': ["balanced"],
              'decision_function_shape': ['ovo', 'ovr']} 
        svm_others = {'C': [0.1, 1, 10],
              'gamma': [1, 0.1, 0.01,'auto'], 
              'kernel': ['poly', 'rbf', 'sigmoid'],
              'class_weight': ["balanced"],
              'decision_function_shape': ['ovo', 'ovr']}
        
        param_grid = [svm_linear, svm_others]
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=uar_scorer, verbose=1)
        
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)
        
        # Display best parameters and best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Validation Accuracy:", grid_search.best_score_)
        
        # Evaluate the best model on the test set
        best_model_val = grid_search.best_estimator_
        best_model_val.fit(X_train, y_train)
        y_pred_val = best_model_val.predict(X_test)
        

        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred_val))
    
    #train and test SVM classifier with constant parameters    
    else:
        print("Fitting SVM for valence with selected parameters")
        
        clf = svm.SVC(kernel="linear", C=1, class_weight="balanced", decision_function_shape="ovo")
        clf.fit(X_train, y_train)
        
        y_pred_val = clf.predict(X_test)
    
    #compute confusion matrix for valence predictions
    val_c_mat = confusion_matrix(y_test, y_pred_val)
    #compute classification metrics for valence
    valence_metrics = classification_scores.get_all_scores(y_test, y_pred_val)
    
    print(valence_metrics)
    
