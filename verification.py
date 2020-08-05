import numpy as np
from statistics import mode
import json



with open("./setting.json") as json_obj:
    mySetting = json.load(json_obj)


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
            
        np.save("./result/Model Evaluation/{}_{}.npy".format(feature_type, distance_type), result)