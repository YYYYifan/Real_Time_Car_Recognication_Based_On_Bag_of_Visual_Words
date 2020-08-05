import numpy as np
import cv2
import datetime
from statistics import mode

k = 3

BoVW_Model = np.load("../result/Key_Points_Description_by_SIFT.npy", allow_pickle=True)
dataset = np.load("../result/dataset/SIFT_Euclidean.npy", allow_pickle=True).item()

SIFT = cv2.xfeatures2d.SIFT_create()     

def cal_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def SIFT_features(image):                                                                     
    key_point, description = SIFT.detectAndCompute(image,None)                           
    return key_point, description


def find_index(each_feature, BoVW_Model):
    count = 0
    ind = 0
    for i in range(len(BoVW_Model)):
        if(i == 0):
           count = np.sqrt(np.sum(np.square(each_feature - BoVW_Model[i]))) 
        else:
            dist = np.sqrt(np.sum(np.square(each_feature - BoVW_Model[i])))
            if(dist < count):
                ind = i
                count = dist
    return ind 
        
def build_dataset(description):                    
    histogram = np.zeros(len(BoVW_Model))
    for each_feature in description:
        ind = find_index(each_feature, BoVW_Model)
        histogram[ind] += 1
    
    return histogram


image = cv2.imread("./Positive.jpg", cv2.IMREAD_GRAYSCALE) 
'''
cv2.imshow("Image", image)
cv2.waitKey (0)  
cv2.destroyAllWindows() 
'''
start_time = datetime.datetime.now()
key_points, description = SIFT_features(image)

histogram = build_dataset(description)

buff = []
for d_key, d_histograms in dataset.items():                        
    for d_histogram in d_histograms:            
        buff.append([cal_distance(histogram, d_histogram), d_key])
        
list.sort(buff)        
k_key = mode(np.asarray(buff[:k])[:, 1])
end_time = datetime.datetime.now()
print(k_key)
print(end_time - start_time)