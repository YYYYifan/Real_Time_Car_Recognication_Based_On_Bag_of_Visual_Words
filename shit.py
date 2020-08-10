from statistics import mode
from PIL import ImageGrab
import numpy as np
import PIL
import cv2
import datetime

BoVW_Model = np.load("./result/Key_Points_Description_by_SIFT.npy", allow_pickle=True)
dataset = np.load("./result/dataset/SIFT_Euclidean.npy", allow_pickle=True).item() 
SIFT = cv2.xfeatures2d.SIFT_create()     
BOX=(0,25,880,640)
 
#左上角坐标和右下角坐标
#调整box的值即可改变截取区域

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

 
while True:    
    start = datetime.datetime.now()
    screen=np.array(ImageGrab.grab(bbox=BOX).convert("L"))
    cv2.imshow("window",screen)
    
    key_points, description = SIFT_features(screen)

    histogram = build_dataset(description)
    
    buff = []
    for d_key, d_histograms in dataset.items():                        
        for d_histogram in d_histograms:            
            buff.append([cal_distance(histogram, d_histogram), d_key])
            
    list.sort(buff)        
    k_key = mode(np.asarray(buff[:3])[:, 1])
    # end_time = datetime.datetime.now()
    print("{}, {}, {}".format(datetime.datetime.now(), k_key, datetime.datetime.now()-start))
    
    
    
    
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break