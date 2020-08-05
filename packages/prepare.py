# -*- coding: utf-8 -*-

import numpy as np
import random
import PIL
import datetime


class imageProcess():
    """
    TEST
    """
    
    def __init__(self, data_path: str, percent: float=0.1, debug: bool=True):
        """
        

        Parameters
        ----------
        root_path : str, optional
            DESCRIPTION. The default is "./data/".
        save_data : bool, optional
            DESCRIPTION. The default is True.
        debug : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.
        """
        self.debug = debug
        self.file_obj = open("./log/prepare.log", "w")      
        self.percent = percent        
        self.func_log("In prepare.imageProcess")                        
        start_time = datetime.datetime.now()  
        self.dict_label_path = np.load(data_path, allow_pickle=True).item()        
        
        self.positive_label_path = self.dict_label_path["81"]   # NOTE: 81 is BMW
        
        self.random_negetive_dataset()
        # self.find_max_car_border()
        self.__len_dict__()
        self.read_image()
        self.split_images()
        self.save_data()
        end_time = datetime.datetime.now()  
        self.func_log("Finish, time cost: {}".format(end_time - start_time))                        
        
        
    def __len__(self):
        """
        Return the number of positive and negetive images

        Returns
        -------
        Int
            the number of positive and negetive images.

        """        
        return len(self.positive_label_path) + self.negetive_label_path
    
    
    def __len_dict__(self):
        """
        Create a dict of data length.     

        Returns
        -------
        dict.
            self.len_dict = {
                "Positive": int number,
                "Negetive": int number
            }

        """
        self.len_dict = {
            "Positive": int(len(self.positive_label_path)),
            "Negetive": int(len(self.negetive_label_path))
            }
        
        return self.len_dict
    
    def random_negetive_dataset(self):
        """
        It is for random choose negetive data from whole dictionnaty without BMW (Positive)        

        Returns
        -------
        None.

        """
        start_time = datetime.datetime.now()        
        self.func_log("\n\tIn random_negetive_dataset()")        
        self.negetive_label_path = []
        for key, paths in self.dict_label_path.items():
            if key != "81":       
                for path in paths:
                    self.negetive_label_path.append(path)
        
        if self.debug: print("    random negetive data")
        random.shuffle(self.negetive_label_path)    # Random negetive label path.
        self.negetive_label_path = self.negetive_label_path[:self.__len_dict__()["Positive"]]
        if self.debug: print("    len(negetive_label_path): {}".format(len(self.negetive_label_path)))
        
        end_time = datetime.datetime.now()            
        self.func_log("\n\t\tTime Cost: {}\n".format(end_time-start_time))
                

    
    
    def read_image(self):      
        """
        This func use background and car border to resize images.
        The reuslt will store in self.positive_images and self.negetive_images

        Returns
        -------
        None.

        """
        start_time = datetime.datetime.now()    
        self.func_log("\n\tIn read_image()")
        self.image_path = []
        self.positive_label_path + self.negetive_label_path
        for path in  self.positive_label_path + self.negetive_label_path:
             self.image_path.append(path.replace("\n", "").replace("label", "image").replace(".txt", ".jpg"))
        
        self.func_log("\t\tCompleted replace label path to image path")
        self.images = []
        for index in range(len(self.image_path)):
            image = PIL.Image.open(self.image_path[index]).convert("L")
            self.images.append(image)
            
        self.func_log("\t\tCompleted resize image")            
        self.func_log("\t\tlen(whole image): {}".format(len(self.images)))            
        self.positive_images = self.images[:self.len_dict["Positive"]]
        self.negetive_images = self.images[ self.len_dict["Negetive"]:]
        end_time = datetime.datetime.now()            
        self.func_log("\n\t\tTime Cost: {}\n".format(end_time-start_time))

    
    def split_images(self):
        """
        Split images to train and verification dataset based on 'precent'.

        Parameters
        ----------
        percent : float, optional
            DESCRIPTION. The default is 0.1. 1 > percent > 0

        Returns
        -------
        None.

        """
        start_time = datetime.datetime.now()    
        self.func_log("\n\tIn split_images()")
        self.split_loc = int(self.len_dict["Positive"] * self.percent)
        self.func_log("\t\tsplit loc: {}".format(self.split_loc))
        self.train = {
            "Positive": self.positive_images[self.split_loc:],
            "Negetive": self.negetive_images[self.split_loc:]
            }        
        self.verification = {
            "Positive": self.positive_images[:self.split_loc],
            "Negetive": self.negetive_images[:self.split_loc]
            }
        self.func_log("\t\ttrain: 'Positive': {}, 'Negetive': {}".format(len(self.train["Positive"]), len(self.train["Negetive"])))
        self.func_log("\t\tverification: 'Positive': {}, 'Negetive': {}".format(len(self.verification["Positive"]), len(self.verification["Negetive"])))
        end_time = datetime.datetime.now()            
        self.func_log("\n\t\tTime Cost: {}\n".format(end_time-start_time))


    def dataset(self):
        pass


        
    def save_data(self):        
        """
        if save_data = TRUE, this func will transfer PIL image to numpy array,
        and save them.

        Returns
        -------
        None.

        """
        if self.debug: print("In save_data()")
        buff_dict = {}
        if self.debug: print("    Transfer train dataset")
        for key, images in self.train.items():
            buff_image = []
            for image in images:
                buff_image.append(np.asarray(image))
            
            buff_dict[key] = buff_image
            
        if self.debug: print("    Save dataset")        
        np.save("./data/dataset.npy", buff_dict)        
        
        buff_dict = {}
        if self.debug: print("    Transfer verification dataset")
        for key, images in self.verification.items():
            buff_image = []
            for image in images:
                buff_image.append(np.asarray(image))
            
            buff_dict[key] = buff_image

        if self.debug: print("    Save sample")
        np.save("./data/sample.npy", buff_dict) 

    def func_log(self, log):
        if self.debug: print(log)    
        self.file_obj.write(log + "\n")         


    
if __name__ == "__main__":    
    myImage = imageProcess(data_path="./data/all_front_view_label_path.npy", save_data=True, debug=True)