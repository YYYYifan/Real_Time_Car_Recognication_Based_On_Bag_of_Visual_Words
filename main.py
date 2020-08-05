# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 19:35:59 2020

@author: duyif
"""

import json
import os
import datetime


from packages import prepare
from packages import BoVW

start_datetime = datetime.datetime.now()

def log_func(log:str):
    print(log)
    file_obj.write(log)
    file_obj.write("\n")

file_obj = open("./log/main.log", "w")

log_func("Main func start, Now: {}\n".format(start_datetime))

with open("./setting.json") as json_obj:
    mySetting = json.load(json_obj)


if not (os.path.exists(mySetting["image_dataset"]) and os.path.exists(mySetting["image_sample"])):    
    prepare.imageProcess(data_path=mySetting["images_path"])

feature_type = 2
distance_type = 1



for images in ["dataset", "sample"]:
    for feature_type in  mySetting["feature_type"]:
        for distance_type in mySetting["distance_type"]:
            log_func("----- images: {}, feature_type: {}, distance_type: {}, NOW: {} -----".format(images, feature_type, distance_type, datetime.datetime.now()))
            myBoVW = BoVW.BoVW(images=images, n_clusters=150, feature_type=feature_type, distance_type=distance_type)


# myBoVW = BoVW.BoVW(images="dataset", n_clusters=150, feature_type=2, distance_type=1)
end_datetime = datetime.datetime.now()
log_func("Finish, time cose: {}".format(end_datetime - start_datetime))            
