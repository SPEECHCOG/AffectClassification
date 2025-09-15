#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 16:30:00 2025

@author: Kalle Lahtinen, kalle.t.lahtinen@tuni.fi

This script implements the functions to execute training and testing for a Support Vector Regression model. 
The expected data is read in from csv-files, that contain the training and testing features as well as the continuous 
sample labels arousal and valence. The script trains two separate 
models for predicting the affect related continuous labels for arousal and valence.

NOTE: The original data is NOT available at this moment. 
The data will be published through Kielipankki.

https://urn.fi/urn:nbn:fi:lb-2025081821 
    
"""

import numpy as np
import sys
import os
import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import classification_scores
from sklearn.metrics import make_scorer

def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    return numerator / denominator


def ccc_for_scorer(y_true, y_pred):
    
    score = concordance_correlation_coefficient(y_true, y_pred)
    
    return score
    
if __name__ == '__main__':
    
    ###########################################################################
    
    #define the root dir for the data
    my_cwd = os.getcwd()
    data_storage_dir = my_cwd+"/data/"
    #continuous or discrete labels, in this script always discrete
    label_type = "continuous"
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
    valence_annotated_tr_labels = valence_tr_val.loc[~valence_tr_val["annotation_mean"].isna(), "a_propagated"]
    valence_GS_labels = valence_gold_std["annotation_mean"]
    
    #arousal training and testing labels
    arousal_annotated_tr_labels = arousal_tr_val.loc[~arousal_tr_val["annotation_mean"].isna(), "a_propagated"]
    arousal_GS_labels = arousal_gold_std["annotation_mean"]
    

    #Cast the data to numpy arrays, z-score normalize (zero mean, unit variance) the data
    X_train = tr_val_features.to_numpy()
    X_test = GS_features.to_numpy()
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    
    # Wrap the custom scoring function
    ccc_scorer = make_scorer(ccc_for_scorer)
    
    
    #SVR training and testing for arousal, 
    #get arousal labels
    y_train = arousal_annotated_tr_labels.to_numpy()
    y_test = arousal_GS_labels.to_numpy()
    
    if grid_search: 
        
        clf = svm.SVR()
        
        svr_linear = {'C': [0.1, 1, 10], 
              'kernel': ['linear']} 
        svr_others = {'C': [0.1, 1, 10],
              'gamma': ['auto', 'scale'], 
              'kernel': ['poly', 'rbf', 'sigmoid']}
        
        param_grid = [svr_linear, svr_others]
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=ccc_scorer, verbose=1)
        
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)
        
        # Display best parameters and best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Validation CCC:", grid_search.best_score_)
        
        # Evaluate the best model on the test set
        best_model_ar = grid_search.best_estimator_
        best_model_ar.fit(X_train, y_train)
        y_pred_ar = best_model_ar.predict(X_test)

        print("\nArousal ccc:\n")
        print(concordance_correlation_coefficient(y_test, y_pred_ar))
    
    
    
    #train and test SVR regressor with constant parameters
    else:
        print("Fitting SVR for arousal with selected parameters")

        clf = svm.SVR(kernel="rbf", C=1)
        clf.fit(X_train, y_train)
        
        y_pred_ar = clf.predict(X_test)
        
        print("Arousal ccc: ")
        print(concordance_correlation_coefficient(y_test, y_pred_ar))
    
    #SVR training and testing for valence
    y_train = valence_annotated_tr_labels.to_numpy()
    y_test = valence_GS_labels.to_numpy()
    
    if grid_search: 
        
        clf = svm.SVR()
        
        svr_linear = {'C': [0.1, 1, 10], 
              'kernel': ['linear']} 
        svr_others = {'C': [0.1, 1, 10],
              'gamma': ['auto', 'scale'], 
              'kernel': ['poly', 'rbf', 'sigmoid']}
        
        param_grid = [svr_linear, svr_others]
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=ccc_scorer, verbose=1)
        
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)
        
        # Display best parameters and best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Validation CCC:", grid_search.best_score_)
        
        # Evaluate the best model on the test set
        best_model_val = grid_search.best_estimator_
        best_model_val.fit(X_train, y_train)
        y_pred_val = best_model_val.predict(X_test)
    
        print("\nValence ccc:\n")
        print(concordance_correlation_coefficient(y_test, y_pred_val))
    
    
    
    #train and test SVR regressor with constant parameters
    else:
        print("Fitting SVR for valence with selected parameters")

        clf = svm.SVR(kernel="rbf", C=10, gamma="auto")
        clf.fit(X_train, y_train)
        
        y_pred_val = clf.predict(X_test)
        
        print("Valence ccc: ")
        print(concordance_correlation_coefficient(y_test, y_pred_val))

