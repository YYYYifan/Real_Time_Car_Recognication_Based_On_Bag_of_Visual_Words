# -*- coding: utf-8 -*-
import numpy as np
from statistics import mode


k = 3

'''
with open('./setting.json', 'r') as file_obj:
    setting = json.load(file_obj)

def json_reader(p1, p2, p3):
    return setting[p1][0][p2][0][p3]
'''

Feature = "SIFT"
Distance = "Manhattan"


for Feature in ["SIFT", "SURF"]:
    for Distance in ["Manhattan", "Euclidean", "Chebyshev", "Cosine"]:
        dataset = np.load("./result/dataset/Dataset_{}_{}.npy".format(Feature, Distance), allow_pickle=True).item()
        sample = np.load("./result/sample/sample_{}_{}.npy".format(Feature, Distance), allow_pickle=True).item()
        
        
        
        result = []
        for k in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
            confusion_martix = np.zeros([2,2], dtype = int) # TN FP / FN TP
            for s_key, s_histograms in sample.items():
                
                for s_histogram in s_histograms:
                    buff = []            
                    for d_key, d_histograms in dataset.items():
                        
                        for d_histogram in d_histograms:            
                            buff.append([np.sum(np.abs(s_histogram - d_histogram)), d_key])
                            
                    list.sort(buff)        
                    k_key = mode(np.asarray(buff[:k])[:, 1])
                    
                    if k_key == s_key:
                        if k_key == "Negetive": confusion_martix[0, 0] = confusion_martix[0, 0] + 1 # TN
                        if k_key == "Positive": confusion_martix[1, 1] = confusion_martix[1, 1] + 1 # TP
                    else:
                        if k_key == "Negetive": confusion_martix[0, 1] = confusion_martix[0, 1] + 1 # FP
                        if k_key == "Positive": confusion_martix[1, 0] = confusion_martix[1, 0] + 1 # FN
                        
            n_sample = np.sum(confusion_martix)
            Accuracy = (confusion_martix[1,1] + confusion_martix[0,0]) / n_sample
            Precision = confusion_martix[1,1] / (confusion_martix[1,1] + confusion_martix[0,1])
            Recall = confusion_martix[1,1] / (confusion_martix[1,1] + confusion_martix[1,0])
            F1_Score = 2 * Precision * Recall /  (Precision + Recall)
            result.append([Accuracy, Precision, Recall, F1_Score])
    
        np.save("./result/Model evaluation/{}_{}.npy".format(Feature, Distance), result)    
        
                        