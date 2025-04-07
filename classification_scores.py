import numpy as np
from sklearn.metrics import confusion_matrix



def get_unweighed_average_recall(y_true, y_pred):
    
    '''
    Recall true_positive / (true_positive + false_negative)
    
    This function implements the unweighed version of recall, 
    meaning that all classes have the same weigh in the calculation. 
    
    '''
    
    cm = confusion_matrix(y_true, y_pred).astype(float)
    
    #calculate the relative (class specific) count of true positives (diagonal) and average over all classes
    for i in range(len(cm)):
        
        row_sum = np.sum(cm[i, :])
        
        cm[i, :] = np.divide(cm[i, :], row_sum)
        
    
    return {"uar": np.mean(np.diagonal(cm))}

def get_recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    tp_total = np.sum(np.diagonal(cm))
    
    fn_total = 0
    
    for i in range(len(cm)):
        
        row = cm[i, :]
        row[i] = 0
        fn_total = fn_total + np.sum(row)
        
    
    return {"recall": (tp_total/(tp_total+fn_total))}
    
    
def get_unweighed_average_precision(y_true, y_pred):
    
    '''
    precision true_positive / (true_positive + false_positive)
    
    This function implements the unweighed version of precision, 
    meaning that all classes have the same weigh in the calculation. 
    
    '''
    
    cm = confusion_matrix(y_true, y_pred).astype(float)
    
    #calculate the relative (class specific) count of true positives (diagonal) and average over all classes
    for i in range(len(cm)):
        
        col_sum = np.sum(cm[:, i])
        
        cm[i, :] = np.divide(cm[:, i], col_sum)
        
    
    return {"uap": np.mean(np.diagonal(cm))}

def get_precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    tp_total = np.sum(np.diagonal(cm))
    
    fp_total = 0
    
    for i in range(len(cm)):
        
        column = cm[:, i]
        column[i] = 0
        fp_total = fp_total + np.sum(column)
        
    
    return {"precision": (tp_total/(tp_total+fp_total))}


def get_accuracy(y_true, y_pred):
    '''
    Accuracy true_positive / all_predictions

    '''
    
    cm = confusion_matrix(y_true, y_pred)
    
    
    return {"accuracy": (np.sum(np.diagonal(cm))/np.sum(cm))}
    
def get_f1_score(y_true, y_pred):
    
    recall = get_recall(y_true, y_pred)["recall"]
    precision = get_precision(y_true, y_pred)["precision"]
    
    return {"f1": (recall+precision)/2}

def get_all_scores(y_true, y_pred):
    
    
    return  (get_unweighed_average_recall(y_true, y_pred),
            get_unweighed_average_precision(y_true, y_pred),
            get_precision(y_true, y_pred),
            get_recall(y_true, y_pred),
            get_accuracy(y_true, y_pred),
            get_f1_score(y_true, y_pred))
    
    
    
    