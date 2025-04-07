#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:30:32 2025

@author: Kalle Lahtinen, kalle.t.lahtinen@tuni.fi

This script uses the ExHuBERT model to predict high and low arousal and high, low and 
neutral valence from raw audio waveforms. The model is downloaded from the Huggingface service
and used as is, without fine-tuning.

NOTE: The data required for running this script (raw audio 3 second windows)
is NOT available at this moment.
    
"""

import numpy as np
import sys
import os
import pandas as pd
import classification_scores
import pickle
import torch
from transformers import AutoModelForAudioClassification
import torch.nn.functional as F
from time import gmtime, strftime
    
def uar_for_scorer(y_true, y_pred):
    
    score = classification_scores.get_unweighed_average_recall(y_true, y_pred)
    
    return score["uar"]


def my_log(logfile, message):
    
    with open(logfile, "a") as my_file:
        my_file.write(message)
        my_file.write("\n")
        my_file.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        my_file.write("\n")
        my_file.write("### \n")

    
if __name__ == '__main__':
    
    ###########################################################################
    
    #define the root dir for the data
    my_cwd = os.getcwd()
    data_storage_dir = my_cwd+"/data/"
    #continuous or discrete labels, in this script always discrete
    label_type = "discrete"
    sbatch_job = 0
        
    output_dir = data_storage_dir+"//"+"exhubert_asis_"+"_"+label_type+"_"+str(sbatch_job)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #define logfile, exhubert jobs run on a clustering server
    logfile = output_dir + "/" + "exhubert_experiment_asis_"+label_type+"_"+".txt"
    
    #read in training, validation and testing subset annnotations for valence and arousal
    my_log(logfile, "Reading in data")   
    valence_gold_std = pd.read_csv(data_storage_dir+"/"+label_type+"_valence_gold_std.csv")
    valence_gold_std = valence_gold_std.set_index("Unnamed: 0")
    valence_tr_val = pd.read_csv(data_storage_dir+"/"+label_type+"_valence_tr_val.csv")
    valence_tr_val = valence_tr_val.set_index("Unnamed: 0")
    arousal_gold_std = pd.read_csv(data_storage_dir+"/"+label_type+"_arousal_gold_std.csv")
    arousal_gold_std = arousal_gold_std.set_index("Unnamed: 0")
    arousal_tr_val = pd.read_csv(data_storage_dir+"/"+label_type+"_arousal_tr_val.csv")
    arousal_tr_val = arousal_tr_val.set_index("Unnamed: 0")
    #exhubert style labels pre calculated for the whole data
    exhubert_labels = pd.read_csv(data_storage_dir+"/"+"exhubert_style_labels_df.csv")
    
    
    #read in exhubert features, i.e. raw utterance audio windowed to 3 second chunks
    #exhubert_features = pd.read_csv(data_storage_dir+"/"+"wav2vec_features.csv")
    #exhubert_features = exhubert_features.set_index("Unnamed: 0")
    f = open(data_storage_dir+"/"+"exhubert_features.npy", 'rb')
    annotated_id_list, exhubert_features = pickle.load(f)
    f.close()
    
    #sanity check for annotation dataframes
    if arousal_gold_std.index.to_list() != valence_gold_std.index.to_list():
        my_log(logfile, "arousal and valence gold std different!")
        sys.exit()
        
    if arousal_tr_val.index.to_list() != valence_tr_val.index.to_list():
        my_log(logfile, "arousal and valence training sets different!")
        sys.exit()
        
    if len(set(arousal_gold_std.index.to_list()).intersection(set(arousal_tr_val.index.to_list()))) > 0:
        my_log(logfile, "Overlapping sample ids in arousal gld std and tr sets")
        sys.exit()
        
    if len(set(valence_gold_std.index.to_list()).intersection(set(valence_tr_val.index.to_list()))) > 0:
        my_log(logfile, "Overlapping sample ids in valence gld std and tr sets")
        sys.exit()
    
    
    
    my_log(logfile, "Separating tr and GS")
    exhubert_tr_labels = exhubert_labels.loc[valence_tr_val["annotation_mode"].dropna().index, "exhubert_labels"]
    exhubert_GS_labels = exhubert_labels["exhubert_labels"].iloc[valence_gold_std.index]
    exhubert_GS_indices = exhubert_GS_labels.index.to_list()

    
    # CONFIG and MODEL SETUPß
    model_name = 'amiriparian/ExHuBERT'
    model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True,
                                                            revision="b158d45ed8578432468f3ab8d46cbe5974380812")
    # Freezing half of the encoder for further transfer learning
    model.freeze_og_encoder()
    
    sampling_rate = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    my_log(logfile, "Device: ")
    my_log(logfile, device.type)
    
    results = {}
    #go through each utterance
    for sample_id in exhubert_GS_indices:
        
        samples = exhubert_features[annotated_id_list.index(sample_id)]
        
        sample_confidences = []
        
        #go through each 3 second window within utterance, predict using
        #model
        for sample in samples:
            
            waveform = torch.from_numpy(sample).float().to(device)
            my_log(logfile, "Predicting "+str(sample_id))
            with torch.no_grad():
                output = model(waveform)
                output = F.softmax(output.logits, dim = 1)
                output = output.detach().cpu().numpy().round(2)
                my_log(logfile, str(np.shape(output)))
                sample_confidences.append(output)
                
        
        #store confidences for each 3 second window
        results[sample_id] = sample_confidences
            
        # [[0.      0.      0.      1.      0.      0.]]
        #          Low          |          High                 Arousal
        # Neg.     Neut.   Pos. |  Neg.    Neut.   Pos          Valence
        # Disgust, Neutral, Kind| Anger, Surprise, Joy          Example emotions

    #store predictions
    outputfile = output_dir + "/" + "exhubert_experiment_asis_"+label_type+"_"+".pickle"
    f = open(outputfile, 'wb')
    pickle.dump(results, f)
    f.close()   