import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 

import json

with open("./setting.json") as file_obj:
    mySetting = json.load(file_obj)
    


for feature_type in mySetting["str_feature_type"]:
    for distance_type in mySetting["str_distance_type"]:                                  
        result = np.load("./result/Model Evaluation/{}_{}.npy".format(feature_type, distance_type), allow_pickle=True) # Accuracy, Precision, Recall, F1_Score
        k = mySetting["k"]
        plt.figure(figsize=(6, 4))        
        plt.title("Model Evaluation for {} and {}".format(feature_type, distance_type))
        plt.xlabel("K's Value")
        plt.ylabel("Precent")
        
        Accuracy, = plt.plot(k, result[:, 0])
        Precision, = plt.plot(k, result[:, 1])
        Recall, = plt.plot(k, result[:, 2])
        F1_Score, = plt.plot(k, result[:, 3])
        
        def to_percent(temp, position):
            return '%1.0f'%(100*temp) + '%'
                
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
        # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                
        plt.xticks(k)
        plt.legend([Accuracy ,Precision, Recall, F1_Score], \
                   ['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        
        plt.savefig("./images/Model_Evaluation/{}_{}.png".format(feature_type, distance_type), dpi = 600)
