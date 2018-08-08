import numpy as np
import operator 
from collections import Counter

def KNN_Classifier(train, train_label, test, K):
    
    euc_distance = np.sum(((train - test) ** 2)) ** 0.5
    
    majority_indices = euc_distance.argsort()[:K]
    
    majority_labels = train_label[majority_indices]
    
    result = Counter(majority_labels).most_common()[0][0]
    
    return result 
