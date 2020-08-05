import numpy as np
from statistics import mode
import json


with open("./setting.json") as json_obj:
    mySetting = json.load(json_obj)

def cal_distance(distance_type, point1, point2):
    if distance_type == "Manhattan":     # Manhattan Distance
        return np.sum(np.abs(point1 - point2))
    
    elif distance_type == "Euclidean":   # Euclidean Distance
        return np.sqrt(np.sum(np.square(point1 - point2)))
    
    elif distance_type == "Chebyshev":   # Chebyshev Distance
        return np.max(np.abs(point1 - point2))         
    
    elif distance_type == "Cosine":   # Cosine Distance
        up = np.dot(point1, point2)           
        down = np.sqrt(np.sum(np.square(point1))) * np.sqrt(np.sum(np.square(point2)))
        return up/down


for feature_type in mySetting["str_feature_type"]:
    for distance_type in mySetting["str_distance_type"]:        
        print("\n --- In {}_{} ---".format(feature_type, distance_type))
        result = []
        dataset = np.load("./result/dataset/{}_{}.npy".format(feature_type, distance_type), allow_pickle=True).item()
        sample = np.load("./result/sample/{}_{}.npy".format(feature_type, distance_type), allow_pickle=True).item()        
        for k in mySetting["k"]:
            print("k=" + str(k))
            confusion_martix = np.zeros([2,2], dtype = int) # TN FP / FN TP
            for s_key, s_histograms in sample.items():            
                for s_histogram in s_histograms:
                    buff = []            
                    for d_key, d_histograms in dataset.items():
                        
                        for d_histogram in d_histograms:            
                            distance = cal_distance(distance_type, s_histogram, d_histogram)
                            buff.append([distance, d_key])
                            
                    list.sort(buff)        
                    k_key = mode(np.asarray(buff[:k])[:, 1])
                    
                    if k_key == s_key:
                        if k_key == "Negetive": confusion_martix[0, 0] = confusion_martix[0, 0] + 1 # TN
                        if k_key == "Positive": confusion_martix[1, 1] = confusion_martix[1, 1] + 1 # TP
                    else:
                        if k_key == "Negetive": confusion_martix[0, 1] = confusion_martix[0, 1] + 1 # FP
                        if k_key == "Positive": confusion_martix[1, 0] = confusion_martix[1, 0] + 1 # FN
                        
            n_sample = np.sum(confusion_martix)
            #if feature_type == "SIFT" and distance_type == "Cosine" and k == 1:
            #    print(confusion_martix)
            Accuracy = (confusion_martix[1,1] + confusion_martix[0,0]) / n_sample
            Precision = confusion_martix[1,1] / (confusion_martix[1,1] + confusion_martix[0,1])
            if confusion_martix[1,1] == 0 or confusion_martix[1,0] == 0:
                Recall = 0
            else:
                Recall = confusion_martix[1,1] / (confusion_martix[1,1] + confusion_martix[1,0])
            if Recall == 0:
                F1_Score = 0
            else:                
                F1_Score = 2 * Precision * Recall /  (Precision + Recall)
            result.append([Accuracy, Precision, Recall, F1_Score])
            
        np.save("./result/Model Evaluation/{}_{}.npy".format(feature_type, distance_type), result)