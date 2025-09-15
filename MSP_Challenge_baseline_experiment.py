#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:30:32 2025

@author: Kalle Lahtinen, kalle.t.lahtinen@tuni.fi

To be used with the Affective Speech Corpus for Spontaneous Finnish created in 2024 / 2025

This script uses the MSP-Baseline model to predict arousal and valence from raw audio waveforms. 
The model is downloaded from the Huggingface service
and used as is, without fine-tuning. This is a dummy example of the experiments.

NOTE: The data required for running this script (raw audio)
is NOT available at this moment. The data will be published through Kielipankki.

https://urn.fi/urn:nbn:fi:lb-2025081821
    
"""

import numpy as np
import sys
import os
import argparse
import pandas as pd
import pickle
import torch

from transformers import AutoModelForAudioClassification
import librosa

from time import gmtime, strftime
    

def my_log(logfile, message):
    
    with open(logfile, "a") as my_file:
        my_file.write(message)
        my_file.write("\n")
        my_file.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        my_file.write("\n")
        my_file.write("### \n")
        
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
   target = "valence"
   
   wavs_dir = "/path/to/wavs/"
   
   #Define environmental variables for huggingface IF NEEDED
   #os.environ["HF_HOME"] = "/path/to/huggingface/home"
   #os.environ["TRANSFORMERS_CACHE"] = "/path/to/transformers/cache"
       
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
    
   #read and normalize continuous annotation scores to match the output range 
   #of the used models
   valence_gs_annotation_mean = valence_gold_std["annotation_mean"].to_numpy()
    
   gs_ids = valence_gold_std.index.to_numpy()
    
   valence_gs_annotation_mean_normalized = (valence_gs_annotation_mean + 1) / 2
    
    
   arousal_gs_annotation_mean = arousal_gold_std["annotation_mean"].to_numpy()
    
   gs_ids = arousal_gold_std.index.to_numpy()
    
   arousal_gs_annotation_mean_normalized = (arousal_gs_annotation_mean + 1) / 2
    
    
    
    #load model
   if target == "valence":
        model = AutoModelForAudioClassification.from_pretrained("3loi/SER-Odyssey-Baseline-WavLM-Valence", trust_remote_code=True)
        gt = valence_gs_annotation_mean_normalized
    
   if target == "arousal":
        model = AutoModelForAudioClassification.from_pretrained("3loi/SER-Odyssey-Baseline-WavLM-Arousal", trust_remote_code=True)
        gt = arousal_gs_annotation_mean_normalized
    
   #get mean/std
   mean = model.config.mean
   std = model.config.std
    
   
   predictions = []
   
   #execute inference one sample at a time, read samples from the wav dir
   for sample_id in gs_ids:
        
        file = str(sample_id)+".wav"
    
        print("Predicting File: "+str(file))
        raw_wav, _ = librosa.load(wavs_dir+file, sr=model.config.sampling_rate)
        
        #normalize the audio by mean/std
        norm_wav = (raw_wav - mean) / (std+0.000001)
        
        #generate the mask
        mask = torch.ones(1, len(norm_wav))
        
        #batch it (add dim)
        wavs = torch.tensor(norm_wav).unsqueeze(0)
        
        
        #predict
        with torch.no_grad():
            pred = model(wavs, mask)
    
        print(model.config.id2label) 
        print(pred)
        predictions.append(pred.item())

    
   #Compute concordance correlation coefficient score    
   CCC = concordance_correlation_coefficient(gt, predictions)