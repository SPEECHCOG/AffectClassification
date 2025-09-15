#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:30:32 2025

@author: Kalle Lahtinen, kalle.t.lahtinen@tuni.fi

To be used with the Affective Speech Corpus for Spontaneous Finnish created in 2024 / 2025

This script implements the functions to execute training and testing for a Support Vector Machine classifier. 
The expected data is read in from csv-files, that can be generated from the original HDF5 dataset using the
HDF5_data_split.py script. 
    
"""

import numpy as np
import sys
import os
import argparse
import pandas as pd

import torch

from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    default_data_collator
)

from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import librosa
import soundfile as sf
from datasets import Dataset, DatasetDict
from time import gmtime, strftime

from sklearn.model_selection import train_test_split


class SERWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config 
        #self.loss_fct = nn.MSELoss()   # since it's regression
        
    def ccc_metric(self, pred, lab):
        m_pred = torch.mean(pred, 0, keepdim=True)
        m_lab = torch.mean(lab, 0, keepdim=True)

        d_pred = pred - m_pred
        d_lab = lab - m_lab

        v_pred = torch.var(pred, 0, unbiased=False)
        v_lab = torch.var(lab, 0, unbiased=False)

        corr = torch.sum(d_pred * d_lab, 0) / (
            torch.sqrt(torch.sum(d_pred ** 2, 0)) * torch.sqrt(torch.sum(d_lab ** 2, 0)) + 1e-8
        )

        s_pred = torch.std(pred, 0, unbiased=False)
        s_lab = torch.std(lab, 0, unbiased=False)

        ccc = (2 * corr * s_pred * s_lab) / (
            v_pred + v_lab + (m_pred[0] - m_lab[0]) ** 2 + 1e-8
        )
        return torch.mean(ccc)   # return scalar CCC

    def forward(self, x=None, mask=None, input_values=None, attention_mask=None, labels=None):
        # call the original model (returns predictions)
        
        if x is None:
            x = input_values
        if mask is None:
            mask = attention_mask
        
        
        pred = self.base_model(x=x, mask=mask)

        loss = None
        if labels is not None:
            ccc = self.ccc_metric(pred, labels)
            loss = 1.0 - ccc
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=pred
        )
    

def my_log(logfile, message):
    
    with open(logfile, "a") as my_file:
        my_file.write(message)
        my_file.write("\n")
        my_file.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        my_file.write("\n")
        my_file.write("### \n")


def load_audio(batch):
    speech, sr = sf.read(batch["path"])
    batch["speech"] = speech
    batch["sampling_rate"] = sr
    return batch

def prepare_features(batch):
    speech = np.array(batch["speech"], dtype=np.float32)
    speech = (speech - mean) / (std + 1e-6)
    inputs = processor(
        speech,
        sampling_rate=batch["sampling_rate"],
        return_tensors="pt",
        padding="max_length",
        max_length=batch["sampling_rate"]*20, #aiemmin oli *20
        truncation=True
    )
    inputs["x"] = inputs.pop("input_values").squeeze(0)
    inputs["mask"] = inputs.pop("attention_mask").squeeze(0)
    inputs["labels"] = float(batch[target])
    return inputs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.squeeze()
    # Concordance Correlation Coefficient (CCC)
    x, y = preds, labels
    x_mean, y_mean = np.mean(x), np.mean(y)
    cov = np.mean((x - x_mean) * (y - y_mean))
    x_var, y_var = np.var(x), np.var(y)
    ccc = (2 * cov) / (x_var + y_var + (x_mean - y_mean)**2 + 1e-8)
    return {"ccc": ccc}


    
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
    sbatch_job = 0
    annotation = label_type
    output_dir = data_storage_dir+"//"+"MSP_baseline_"+annotation+"_"+label_type+"_"+str(target)+"_"+str(sbatch_job)
    
    wavs_dir = "/path/to/wavs/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logfile = output_dir + "/" + "MSP_experiment_finetune_"+label_type+"_"+annotation+"_"+target+".txt"
    
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
     
    arousal_gs_annotation_mean = arousal_gold_std["annotation_mean"].to_numpy()
    gs_ids = arousal_gold_std.index.to_numpy()
     
    # ---------------------------
    # 1. Load labels and split to training, validation and testing (GS)
    # ---------------------------
    
    if target == "valence":
    
        df = valence_tr_val
        df_index = df.index.to_list()
        df["sample_ids"] = df_index
        df = valence_tr_val[["sample_ids", "annotation_mean"]]
        
        df_testing = valence_gold_std
        df_testing["valence"] = (df_testing["annotation_mean"] + 1) / 2
        df_testing["sample_ids"] = gs_ids
        
        df_testing = df_testing[["sample_ids", "annotation_mean", "valence"]]
    
    if target == "arousal":
        
       df = arousal_tr_val
       df_index = df.index.to_list()
       df["sample_ids"] = df_index
       df = arousal_tr_val[["sample_ids", "annotation_mean"]]
       
       df_testing = arousal_gold_std
       df_testing["arousal"] = (df_testing["annotation_mean"] + 1) / 2
       df_testing["sample_ids"] = gs_ids
       df_testing = df_testing[["sample_ids", "annotation_mean", "arousal"]]
    

    df.dropna(inplace=True, subset=["annotation_mean"])
    df["path"] = df["sample_ids"].apply(lambda x: os.path.join(wavs_dir, str(x)+".wav"))
    df_testing["path"] = df_testing["sample_ids"].apply(lambda x: os.path.join(wavs_dir, str(x)+".wav"))
    
    if target == "valence":
    
        df["valence"] = (df["annotation_mean"] + 1) / 2
        
    if target == "arousal":
    
        df["arousal"] = (df["annotation_mean"] + 1) / 2
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))
    tst_ds = Dataset.from_pandas(df_testing.reset_index(drop=True))
    dataset  = DatasetDict({"train": train_ds, "validation": val_ds, "test": tst_ds})
    
    # ---------------------------
    # 2. Load model & processor
    # ---------------------------
    
    if target == "valence":
    
        model_name = "3loi/SER-Odyssey-Baseline-WavLM-Valence"
        
    if target == "arousal":
    
        model_name = "3loi/SER-Odyssey-Baseline-WavLM-Arousal"
    
    model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
    model = SERWrapper(model)
    
    from transformers import Wav2Vec2FeatureExtractor
    processor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=False,
        return_attention_mask=True,
    )
    
    # Normalization values from model config
    mean, std = model.config.mean, model.config.std
    
    # ---------------------------
    # 3. Preprocessing
    # ---------------------------
    
    dataset = dataset.map(load_audio)
    dataset = dataset.map(
        prepare_features,
        remove_columns=["annotation_mean", "sample_ids", "path", "speech", "sampling_rate", target]
    )
    dataset.set_format(type="torch")
    
    # ---------------------------
    # 4. Training setup
    # ---------------------------
    
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/finetuned-{target}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        learning_rate=1e-5,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        logging_dir="./logs",      # where to save TensorBoard logs
        logging_strategy="steps",  # log every X steps
        log_level="info",   
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="ccc",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=False,   # <-- crucial
        disable_tqdm=True
    )
    
    data_collator = default_data_collator
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # ---------------------------
    # 5. Train
    # ---------------------------
    print("Starting training")
    print("Trainer is using device:", trainer.args.device)
    my_log(logfile,"STARTING TRAINING")
    my_log(logfile,"Trainer is using device: "+str(trainer.args.device))
    trainer.train()
    
    # ---------------------------
    # 6. Evaluate
    # ---------------------------
    print("Starting evaluation")
    eval_res = trainer.evaluate(eval_dataset=dataset["test"])
    my_log(logfile,"STARTING TESTING")
    my_log(logfile,"Evaluation results:")
    my_log(logfile, str(eval_res))
    
    
    
    
    