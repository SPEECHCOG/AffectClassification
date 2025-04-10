# AffectClassification

The python code files and data for running affect classification experiments on spontaneous speech features and data. There are three different experiments implemented. Unpack the data.zip, all the available data should then be in a directory called "data".

The data used for SVM experiments consists of eGeMAPSv02 features calculated using the original raw audio, previously calculated SER posteriors using the original raw audio and text sentiment posteriors using the original audio transcripts. 

The data used for ExHuBERT experiments (classification with and without fine-tuning) should be conducted using raw audio samples, which are not shared at this moment because of data licensing reasons (we're working on it!). The data shared here at this moment (7.4.2025) is dummy data with the same array-structure as the actual data. The content is normally distributed noise. 

# SVM Experiment

File: SVM_experiment.py
Execution: python SVM_experiment.py 

No commandline arguments needed. You can exeute a grid search for different SVM parameters by changing the variable value in the beginning of the script. If grid_search is False, the code just trains and tests SVM classifiers using constant (best performing) parameters. 

# ExHuBERT Experiment

File: ExHuBERT_experiment.py
Execution: python ExHuBERT_experiment.py

No commandline arguments needed. The script downloads the model from Huggingface and runs classification using the provided data. 


# ExHuBERT Experiment with fine-tuning

File: ExHuBERT_experiment_finetune.py
Execution: python ExHuBERT_experiment_finetune.py

No commandline arguments needed. The script downloads the model from Huggingface, splits the training data into training and validation. The model classifier layer is retrained (transfer learning) for the new domain. The best performing models (as measured in the validation loop during training) are stored and finally used for testing with a separate test data set. 

