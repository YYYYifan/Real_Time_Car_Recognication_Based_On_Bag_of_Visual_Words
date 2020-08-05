# -*- coding: utf-8 -*-
import cv2
import numpy as np
import datetime
from sklearn.cluster import KMeans
import os
import json


class BoVW():
    """
    This class is used for build bag of viusal words model.
    """
    def __init__(self, images: str="dataset", n_clusters: int=150, feature_type: int=1, distance_type: int=1, debug: bool=True):
        """
        

        Parameters
        ----------
        train_images : dict
            images{
                "c1": [PIL Image, PIL Image, .....],
                "c2": [PIL Image, PIL Image, .....]                
                }.
        n_clusters : int, optional
            This one decide how many clusters in k-means. 
            The default is 150.
        feature_type : int, optional
            Select the way to extract feature from image , 1 is SIFT and 2 is SURF. 
            The default is 1.
        distance_type : int, optional
            Select the way to calculate distance.
                1. Manhattan Distance
                2. Euclidean Distance
                3. Chebyshev Distance
                4. Cosine Distance
            The default is 1.
        debug : bool, optional
            Whether or not show basically information in console. The default is True.

        Returns
        -------
        None.

        """
        self.debug = debug        
        
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
        
        self.title = {
            "log": "{}/Extract_{}, Distance_{}.log".format(images, dict_feature_name, dict_distance_name),
            "BoVW": "/Key_Points_Description_by_{}.npy".format(dict_feature_name),
            "Key_Points": "{}/Key_Points_by_{}.npy".format(images, dict_feature_name),
            "Descriptor_List": "{}/Descriptor_List_by_{}.npy".format(images, dict_feature_name),
            "Dataset": "{}/{}_{}.npy".format(images, dict_feature_name, dict_distance_name)
            }
        
        
        with open("./setting.json") as file_obj:
            self.mySetting = json.load(file_obj)
        
        if images == "dataset":
            self.images = np.load(self.mySetting["image_dataset"], allow_pickle=True).item()
        elif images == "sample":
            self.images = np.load(self.mySetting["image_sample"], allow_pickle=True).item()
            
        self.file_obj = open(self.mySetting["BoVW"][0]["log_path"] + self.title["log"], "w")       
        self.func_log("In packages.BoVW")
        start_time = datetime.datetime.now()      
        # for self.images in [image_dataset, image_sample]:
        
        self.distance_type = distance_type
                                                        
        # --- Start        
        
        # --- Features
        # NOTE: Fix train and verification.        
        if not (os.path.exists("./data/{}".format(self.title["Descriptor_List"])) and (os.path.exists("./data/{}".format(self.title["Key_Points"])))):
            
            self.func_log("\tFeatures doesn`t exist.")            
            if feature_type == 1:
                self.SIFT_features()
            elif feature_type == 2:
                self.SURF_features()
            self.save_data("./data/{}".format(self.title["Descriptor_List"]), self.descriptor_list)    
            self.save_data("./data/{}".format(self.title["Key_Points"]), self.key_points)
        else:
            self.func_log("\tFound the features.")                        
            self.descriptor_list = np.load("./data/{}".format(self.title["Descriptor_List"]), allow_pickle=True)
            self.key_points = np.load("./data/{}".format(self.title["Key_Points"]), allow_pickle=True).item()
                                                  
        # K-Means needs a lot of time, so this program will check if there has a BoVW model firstlly.
        # --- BoVW Models
        
        if not os.path.exists("./result/{}".format(self.title["BoVW"])):
            self.func_log("\tBoVW model doesn`t exist")
            self.K_Means(n_clusters)            
            self.save_data("./result/{}".format(self.title["BoVW"]), self.visual_words)    
        else:
            self.func_log("\tFound the BoVW model")            
            self.visual_words = np.load("./result/{}".format(self.title["BoVW"]), allow_pickle=True)        
        
        # --- BoVW dataset        
        if not os.path.exists("./result/dataset/{}".format(self.title["Dataset"])):            
            self.func_log("\tBoVW dataset doesn`t exist")
            self.build_dataset()
            self.save_data("./result/{}".format(self.title["Dataset"]), self.dict_feature)    
        else:    
            self.func_log("\tFound the BoVW dataset.")            
            self.key_points = np.load("./data/{}".format(self.title["Dataset"]), allow_pickle=True).item()
        
        
        #self.save_data()                
        
        end_time = datetime.datetime.now()            
        self.func_log("BoVW dataset is ready, time cose: {}".format(end_time-start_time))                
        self.file_obj.close()    

    
    def SIFT_features(self):
        """
        Extraction feature by Scale-invariant feature transform(SIFT).

        Returns
        -------
        None.

        """
        start_time = datetime.datetime.now()        
        self.func_log("\n\tIn SIFT_features()")
            
        key_points = {}
        descriptor_list = []
        SIFT = cv2.xfeatures2d.SIFT_create()
                
        self.func_log("\t\tSIFT feature extraction start")
            
        for key, value in self.images.items():
            features = []            
            for img in value:
                kp, des = SIFT.detectAndCompute(img,None)                           
                descriptor_list.extend(des)
                features.append(des)
            
            key_points[key] = features    
            
            self.func_log("\t\t\tKEY: {} finished".format(key))
                            
        self.descriptor_list = descriptor_list
        self.key_points = key_points    
        
        end_time = datetime.datetime.now()            
        self.func_log("\n\t\tTime Cost: {}\n".format(end_time-start_time))
        
    
    def SURF_features(self):
        """
        Extraction feature by Speeded Up Robust Features(SURF).

        Returns
        -------
        None.

        """
        start_time = datetime.datetime.now()
        self.func_log("\n\tIn SURF_features()")
            
        key_points = {}
        descriptor_list = []
        surf = cv2.xfeatures2d.SURF_create()
        
        self.func_log("\t\tSURF feature extraction start")
        
        for key, value in self.images.items():
            features = []            
            for img in value:
                kp, des = surf.detectAndCompute(img,None)                           
                descriptor_list.extend(des)
                features.append(des)
            
            key_points[key] = features
            
            self.func_log("\t\t\tKEY: {} finished".format(key))
                   
        self.descriptor_list = descriptor_list
        self.key_points = key_points
        
        end_time = datetime.datetime.now()            
        self.func_log("\n\t\tTime Cost: {}\n".format(end_time-start_time))
            
        
    def K_Means(self, n_clusters: int=150):
        """
        K-Means by using key point description.

        Parameters
        ----------
        n_clusters : int
            How many clusters in K-Means.

        Returns
        -------
        None.

        """
        start_time = datetime.datetime.now()
        self.func_log("\n\tIn K-Measn()")
            
        kmeans = KMeans(n_clusters = n_clusters)
        kmeans.fit(self.descriptor_list)
        self.visual_words = kmeans.cluster_centers_ 
        
        end_time = datetime.datetime.now()            
        self.func_log("\n\t\tTime Cost: {}\n".format(end_time-start_time))       

            
    def cal_distance(self, point1, point2):
        """
        Calculate distance.

        Parameters
        ----------
        point1 : TYPE
            DESCRIPTION.
        point2 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.distance_type == 1:     # Manhattan Distance
            return np.sum(np.abs(point1 - point2))
        
        elif self.distance_type == 2:   # Euclidean Distance
            return np.sqrt(np.sum(np.square(point1 - point2)))
        
        elif self.distance_type == 3:   # Chebyshev Distance
            return np.max(np.abs(point1 - point2))         
        
        elif self.distance_type == 4:   # Cosine Distance
            up = np.dot(point1, point2)           
            down = np.sqrt(np.sum(np.square(point1))) * np.sqrt(np.sum(np.square(point2)))
            return up/down
        
        
    def build_dataset(self):
        """
        Build histgram, then make it to dataset(K-NN).        

        Returns
        -------
        None.

        """        
        start_time = datetime.datetime.now()
        self.func_log("\n\tIn build_dataset()")
        
        self.dict_feature = {}
        for key,value in self.key_points.items():
            category = []
            buff_time = datetime.datetime.now()
            for img in value:
                histogram = np.zeros(len(self.visual_words))
                for each_feature in img:
                    ind = self.find_index(each_feature, self.visual_words)
                    histogram[ind] += 1
                category.append(histogram)
            self.dict_feature[key] = category
            
            buff_time = datetime.datetime.now() - buff_time
            self.func_log("\t\tKEY: {} finish, Time cose:{}".format(key, buff_time))
        end_time = datetime.datetime.now()            
        self.func_log("\n\t\tTime Cost: {}\n".format(end_time-start_time))              
        
    
    
    def find_index(self, each_feature, model):
        count = 0
        ind = 0
        for i in range(len(model)):
            if(i == 0):
               count = self.cal_distance(each_feature, model[i]) 
               #count = L1_dist(image, center[i])
            else:
                dist = self.cal_distance(each_feature, model[i]) 
                #dist = L1_dist(image, center[i])
                if(dist < count):
                    ind = i
                    count = dist
        return ind    
    
        
    def save_data(self, path, data):
        np.save(path, data)
            
    
    def func_log(self, log):
        if self.debug: print(log)    
        self.file_obj.write(log + "\n") 
        