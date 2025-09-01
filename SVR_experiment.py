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
    
    
    #SVM training and testing for arousal, 
    #get arousal labels
    y_train = arousal_annotated_tr_labels.to_numpy()
    y_test = arousal_GS_labels.to_numpy()
    
    if grid_search: 
        regressors = {'svr_linear': svm.SVR(kernel='linear'),
                      'svr_rbf': svm.SVR(kernel='rbf'),
                      'svr_poly': svm.SVR(kernel='poly')}
        
        for regr_name, regr in regressors.items():
            regr.fit(X_train, y_train)
            
            y_pred = regr.predict(X_test)
            
            print(regr_name+" arousal scores: ")
            #print(mean_squared_error(y_test, y_pred))
            #print(mean_absolute_error(y_test, y_pred))
            #print(r2_score(y_test, y_pred))
            print(concordance_correlation_coefficient(y_test, y_pred))
    
    
    
    else:
    
        regr = svm.SVR(kernel='rbf')
        regr.fit(X_train, y_train)
        
        y_pred = regr.predict(X_test)
        
        print("Arousal scores: ")
        print(concordance_correlation_coefficient(y_test, y_pred))
    
    #SVM training and testing for valence
    y_train = valence_annotated_tr_labels.to_numpy()
    y_test = valence_GS_labels.to_numpy()
    
    if grid_search: 
        regressors = {'svr_linear': svm.SVR(kernel='linear'),
                      'svr_rbf': svm.SVR(kernel='rbf'),
                      'svr_poly': svm.SVR(kernel='poly')}
        
        for regr_name, regr in regressors.items():
            regr.fit(X_train, y_train)
            
            y_pred = regr.predict(X_test)
            
            print(regr_name+" valence scores: ")
            #print(mean_squared_error(y_test, y_pred))
            #print(mean_absolute_error(y_test, y_pred))
            #print(r2_score(y_test, y_pred))
            print(concordance_correlation_coefficient(y_test, y_pred))
    
    
    else:
    
        regr = svm.SVR(kernel='linear')
        regr.fit(X_train, y_train)
        
        y_pred = regr.predict(X_test)
        
        print("Valence scores: ")
        print(concordance_correlation_coefficient(y_test, y_pred))

