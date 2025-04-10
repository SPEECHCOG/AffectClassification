#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:30:32 2025

@author: Kalle Lahtinen, kalle.t.lahtinen@tuni.fi

This script uses the ExHuBERT model to predict high and low arousal and high, low and 
neutral valence from raw audio waveforms. The model is downloaded from the Huggingface service
and used for first fine-tuning the model for predicting affect from spontaneous speech data and
finally testing the best performing models (for arousal and valence separately) with the 
gold standard dataset.

NOTE: The data required for running this script (raw audio 3 second windows)
is NOT available at this moment.
    
"""

import numpy as np
import sys
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import classification_scores
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification
from time import gmtime, strftime
from torch.utils.data import Dataset, DataLoader

def custom_collate(batch):
    waveforms, labels = zip(*batch)  # Separate inputs & labels
    waveforms = torch.stack(waveforms)  # Stacks into (batch_size, 48000)
    labels = torch.stack(labels)  # Stacks into (batch_size, 6)
    return waveforms, labels

# Define a custom dataset class
class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.data = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = self.data[idx].squeeze(0)
        label = self.labels[idx]

        return waveform, label
    
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
        
def exhubert_predictions_to_valence_arousal(predictions):
    
    arousal_predicted_labels = []
    valence_predicted_labels = []
    exhubert_predicted_labels = []
    
    for prediction in predictions:
        
   
        exhubert_label = np.argmax(prediction[0])
        
        exhubert_predicted_labels.append(exhubert_label)
        
        if exhubert_label == 0:
            
            arousal_predicted_labels.append(0)
            valence_predicted_labels.append(0)
        
        if exhubert_label == 1:
            
            arousal_predicted_labels.append(0)
            valence_predicted_labels.append(1)
            
        if exhubert_label == 2:
            
            arousal_predicted_labels.append(0)
            valence_predicted_labels.append(2)
            
        if exhubert_label == 3:
            
            arousal_predicted_labels.append(1)
            valence_predicted_labels.append(0)
            
        if exhubert_label == 4:
            
            arousal_predicted_labels.append(1)
            valence_predicted_labels.append(1)
            
        if exhubert_label == 5:
            
            arousal_predicted_labels.append(1)
            valence_predicted_labels.append(2)
            
            
    return arousal_predicted_labels, valence_predicted_labels, exhubert_predicted_labels

    
if __name__ == '__main__':
    
    ###########################################################################
    
        
    #define the root dir for the data
    my_cwd = os.getcwd()
    data_storage_dir = my_cwd+"/data/"
    #continuous or discrete labels, in this script always discrete
    label_type = "discrete"
    sbatch_job = 0
    num_epochs = 10
        
    output_dir = data_storage_dir+"//"+"exhubert_finetuned_"+"_"+label_type+"_"+str(sbatch_job)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logfile = output_dir + "/" + "exhubert_experiment_finetuned_"+label_type+"_"+".txt"
        
    my_log(logfile, "Reading in data") 
    #read in training, validation and testing subset annnotations for valence and arousal
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
    #NOTE: THE DATA THAT IS BEING READ HERE IS NOT REAL AUDIO, 
    #BUT NORMALLY DISTRIBUTED NOISE
    f = open(data_storage_dir+"/"+"exhubert_DUMMY_features_noise.npy", 'rb')
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
    exhubert_tr_labels_series = exhubert_labels.loc[valence_tr_val["annotation_mode"].dropna().index, "exhubert_labels"]
    exhubert_GS_labels_series = exhubert_labels["exhubert_labels"].iloc[valence_gold_std.index]
    exhubert_GS_indices = exhubert_GS_labels_series.index.to_list()
    exhubert_tr_indices = exhubert_tr_labels_series.index.to_list()
    
    exhubert_tr_features = []
    exhubert_tr_labels = []
    exhubert_tr_labels_og = []
    exhubert_GS_features = []
    exhubert_GS_labels = []
    
    
    for sample_id in exhubert_tr_indices:
        
        sample_features = exhubert_features[annotated_id_list.index(sample_id)]
        sample_label = exhubert_tr_labels_series[sample_id]
        sample_label_onehot = np.zeros(6)
        
        sample_label_onehot[int(sample_label)] = 1
        for sample_feature in sample_features:
            
            exhubert_tr_features.append(sample_feature)
            
            exhubert_tr_labels.append(sample_label_onehot)
            exhubert_tr_labels_og.append(sample_label)
            
          
    for sample_id in exhubert_GS_indices:
        
        sample_features = exhubert_features[annotated_id_list.index(sample_id)]
        sample_label = exhubert_GS_labels_series[sample_id]
        
        
        for sample_feature in sample_features:
            
            exhubert_GS_features.append(sample_feature)
            exhubert_GS_labels.append(sample_label)
    

    
    #split the training data into training and validation splits
    #define datasets and dataloaders
    x_train, x_test, y_train, y_test = train_test_split(exhubert_tr_features, exhubert_tr_labels, test_size=0.2)
    
    tr_dataset = EmotionDataset(x_train, y_train)
    tr_dataloader = DataLoader(tr_dataset, batch_size=100, shuffle=True, collate_fn=custom_collate, num_workers=0)
    
    val_dataset = EmotionDataset(x_test, y_test)
    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate)
    


    # CONFIG and MODEL SETUPß
    model_name = 'amiriparian/ExHuBERT'
    model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True,
                                                            revision="b158d45ed8578432468f3ab8d46cbe5974380812")
    # Training setup
    criterion = nn.CrossEntropyLoss()
    lr = 1e-5
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(non_frozen_parameters, lr=lr, betas=(0.9, 0.999), eps=1e-08)
    
    # Replacing Classifier layer
    model.classifier = nn.Linear(in_features=256, out_features=6)
    # Freezing the original encoder layers and feature encoder (as in the paper) for further transfer learning
    model.freeze_og_encoder()
    model.freeze_feature_encoder()

    if sys.platform == "darwin":
        print("OS X operating system")
        device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    else: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    sampling_rate = 16000
    model = model.to(device)
    
    my_log(logfile, "Device: ")
    my_log(logfile, device.type)
    
    results = {}
    
    valence_best_model = {}
    arousal_best_model = {}
    exhubert_best_model = {}
    
    best_models = {}
    best_models["valence"] = None
    best_models["arousal"] = None
    best_models["exhubert"] = None
    
    epoch_losses = []
    epoch_uars = []
    
    #training and validation loop
    for epoch in range(num_epochs):
        
        my_log(logfile, "Training, epoch "+str(epoch))
        
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        # Training loop
        for batch_idx, (inputs, targets) in enumerate(tr_dataloader):
            my_log(logfile, "Training, batch "+str(batch_idx))
            my_log(logfile, "Training, epoch "+str(epoch))
            
            inputs, targets = inputs.to(device), targets.to(device)
    
            optim.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, targets)
            epoch_losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optim.step()
            
        
        my_log(logfile, "Validating, epoch "+str(epoch))
        # Validation, store the model if it performs better than the 
        # last best model for both arousal and valence separately
        model.eval()
        total_loss, correct, total = 0, 0, 0
        
        #separate arousal and valence labels from the exbubert posterior
        predicted_arousals = []
        predicted_valences = []
        predicted_exhuberts = []
        sample_confidences = []
        sample_GTs = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_dataloader):                
                my_log(logfile, "Validating, batch "+str(batch_idx))
                inputs = inputs.to(device)
                sample_GTs.append(targets)
                output = model(inputs)
                output = F.softmax(output.logits, dim = 1)
                output = output.detach().cpu().numpy().round(2)
                sample_confidences.append(output)
            
        
        #get separate labels from exhubert posteriors
        arousal_predicted_labels, valence_predicted_labels, exhubert_predicted_labels = exhubert_predictions_to_valence_arousal(sample_confidences)
        arousal_GT_labels, valence_GT_labels, exhubert_GT_labels = exhubert_predictions_to_valence_arousal(sample_GTs)
        
        #compute valence confusion matrix and UAR from validation predictions
        my_log(logfile, "Valence metrics")
        
        val_c_mat = confusion_matrix(valence_GT_labels, valence_predicted_labels)
        
        valence_metrics = classification_scores.get_all_scores(valence_GT_labels, valence_predicted_labels)
        
        my_log(logfile, str(valence_metrics[0]["uar"]))
        
        if best_models["valence"] == None:
            my_log(logfile, "First UAR for valence "+str(valence_metrics[0]["uar"]))
            best_models["valence"] = (str(epoch), model.state_dict(), optim.state_dict(), valence_metrics[0]["uar"])
            
        else:
            
            if best_models["valence"][3] < valence_metrics[0]["uar"]:
                
                my_log(logfile, "Improved UAR for valence "+str(valence_metrics[0]["uar"]))
                
                best_models["valence"] = (str(epoch), model.state_dict(), optim.state_dict(), valence_metrics[0]["uar"])
                
        #compute arousal confusion matrix and UAR from validation predictions
        my_log(logfile, "Arousal metrics")
        
        ar_c_mat = confusion_matrix(arousal_GT_labels, arousal_predicted_labels)
        
        arousal_metrics = classification_scores.get_all_scores(arousal_GT_labels, arousal_predicted_labels)
        
        my_log(logfile, str(arousal_metrics[0]["uar"]))
        
        
        if best_models["arousal"] == None:
            my_log(logfile, "First UAR for arousal "+str(arousal_metrics[0]["uar"]))
            best_models["arousal"] = (str(epoch), model.state_dict(), optim.state_dict(), arousal_metrics[0]["uar"])
            
        else:
            
            if best_models["arousal"][3] < arousal_metrics[0]["uar"]:
                
                my_log(logfile, "Improved UAR for arousal "+str(arousal_metrics[0]["uar"]))
                
                best_models["arousal"] = (str(epoch), model.state_dict(), optim.state_dict(), arousal_metrics[0]["uar"])
        
        
        #compute exhubert confusion matrix and UAR from validation predictions
        my_log(logfile, "Exhubert metrics")
        
        ex_c_mat = confusion_matrix(exhubert_GT_labels, exhubert_predicted_labels)
        
        exhubert_metrics = classification_scores.get_all_scores(exhubert_GT_labels, exhubert_predicted_labels)
        
        my_log(logfile, str(exhubert_metrics[0]["uar"]))
        
        
        if best_models["exhubert"] == None:
            my_log(logfile, "First UAR for exhubert "+str(exhubert_metrics[0]["uar"]))
            best_models["exhubert"] = (str(epoch), model.state_dict(), optim.state_dict(), exhubert_metrics[0]["uar"])
            
        else:
            
            if best_models["exhubert"][3] < exhubert_metrics[0]["uar"]:
                
                my_log(logfile, "Improved UAR for exhubert "+str(exhubert_metrics[0]["uar"]))
                
                best_models["exhubert"] = (str(epoch), model.state_dict(), optim.state_dict(), exhubert_metrics[0]["uar"])
    

        epoch_uars.append((valence_metrics[0]["uar"], arousal_metrics[0]["uar"], exhubert_metrics[0]["uar"]))

    model_outputfile = output_dir + "/" + "exhubert_experiment_finetuned_"+label_type+"_best_models.pth"
    torch.save(best_models, model_outputfile)
    
    
    #Test the best performing models against the gold standard set
    #Calculate UAR for valence, arousal and exhubert-style labels 
    #separately
    
    results_valence = {}
    results_arousal = {}
    results_exhubert = {}
    
    
    for model_name in best_models.keys():
        
        my_log(logfile, "Loading state dict and testing "+str(model_name))
        
        model.load_state_dict(best_models[model_name][1])
    
        for sample_id in exhubert_GS_indices:
            
            samples = exhubert_features[annotated_id_list.index(sample_id)]
            sample_confidences = []
        
            for sample in samples:
                
                #sample = np.reshape(sample, (1, len(sample)))
                
                waveform = torch.from_numpy(sample).float().to(device)
                my_log(logfile, "Model "+str(model_name))
                my_log(logfile, "Predicting "+str(sample_id))
                with torch.no_grad():
                    output = model(waveform)
                    output = F.softmax(output.logits, dim = 1)
                    output = output.detach().cpu().numpy().round(2)
                    my_log(logfile, str(np.shape(output)))
                    sample_confidences.append(output)
                    
            if model_name == "valence":
            
                results_valence[sample_id] = sample_confidences
                
            if model_name == "arousal":
            
                results_arousal[sample_id] = sample_confidences
                
            if model_name == "exhubert":
            
                results_exhubert[sample_id] = sample_confidences
            
        # [[0.      0.      0.      1.      0.      0.]]
        #          Low          |          High                 Arousal
        # Neg.     Neut.   Pos. |  Neg.    Neut.   Pos          Valence
        # Disgust, Neutral, Kind| Anger, Surprise, Joy          Example emotions


    outputfile = output_dir + "/" + "exhubert_experiment_finetuned_"+label_type+"_epoch_losses.pickle"
    f = open(outputfile, 'wb')
    pickle.dump(epoch_losses, f)
    f.close()
    
    outputfile = output_dir + "/" + "exhubert_experiment_finetuned_"+label_type+"_epoch_uars.pickle"
    f = open(outputfile, 'wb')
    pickle.dump(epoch_uars, f)
    f.close()

    outputfile = output_dir + "/" + "exhubert_experiment_finetuned_"+label_type+"_valence.pickle"
    f = open(outputfile, 'wb')
    pickle.dump(results_valence, f)
    f.close()
    
    outputfile = output_dir + "/" + "exhubert_experiment_finetuned_"+label_type+"_arousal.pickle"
    f = open(outputfile, 'wb')
    pickle.dump(results_arousal, f)
    f.close()   
    
    outputfile = output_dir + "/" + "exhubert_experiment_finetuned_"+label_type+"_exhubert.pickle"
    f = open(outputfile, 'wb')
    pickle.dump(results_exhubert, f)
    f.close()   


'''
    outputfile = output_dir + "/" + "exhubert_experiment_finetuned_"+label_type+".pickle"
    f = open(outputfile, 'wb')
    pickle.dump(results, f)
    f.close()   
'''
