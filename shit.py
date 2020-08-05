# -*- coding: utf-8 -*-

import json
import numpy as np

with open("./setting.json") as file_obj:
    mySetting = json.load(file_obj)

# tt = np.load("./data/dataset/Extract_Feature_by_SIFT.npy", allow_pickle=True).item()

tt = mySetting["distance_type"]

#ttt = np.load("./data/dataset/De_by_SIFT.npy", allow_pickle=True) # Key_Points

# key_points = np.load("./data/dataset/Key_Points_by_SIFT.npy", allow_pickle=True).item()
'''
dataset = np.load("./result/dataset/SIFT_Manhattan.npy", allow_pickle=True).item()
sample = np.load("./result/sample/SIFT_Manhattan.npy", allow_pickle=True).item()


feature_type = 1
distance_type = 1

dict_feature_name = {
    1: "SIFT",
    2: "SURF"
    }[feature_type]
        
dict_distance_name = {
    1: "Manhattan",
    2: "Euclidean",
    3: "Chebyshev",
    4: "Cosine"
    }[distance_type]  

obj_name = "{}_{}.npy".format(dict_feature_name, dict_distance_name)


dataset = np.load("./result/dataset/{}_{}.npy".format(dict_feature_name, dict_distance_name), allow_pickle=True).item()
sample = np.load("./result/sample/{}_{}.npy".format(dict_feature_name, dict_distance_name), allow_pickle=True).item()
'''

for feature_type in mySetting["str_feature_type"]:
    for distance_type in mySetting["str_distance_type"]:  
        result = np.load("./result/Model Evaluation/{}_{}.npy".format(feature_type, distance_type), allow_pickle=True)
        print("./result/Model Evaluation/{}_{}.npy, len = {}\n".format(feature_type, distance_type, len(result[:, 0])))
