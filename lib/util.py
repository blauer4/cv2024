import numpy as np
import json
import datetime
import cv2 
import os

def create2DArray(start_val, end_val):
    chess_2d_array = []
    val = np.arange(start_val,end_val)
    for a in val:
        for b in val:
            chess_2d_array.append([a,b])
    return chess_2d_array

#corners: quello che ti dà la findChessboardCorners
def saveToJSON(dict: dict, camera_number: int):
    with open(f"{camera_number}corners.json", "a+") as outfile:
        json.dump(dict, outfile)
        return True
    return False

def getMilliSeconds(dat: datetime):
    t = dat.time()
    return (t.hour * 60 * 60 * 1000) + (t.minute * 60 * 1000) + (t.second * 1000) + (t.microsecond / 1000) 

def LoadJSON(filepath) -> dict:
    with open(filepath, "r") as outfile:
            return json.load(outfile)

def showImage(img, window_name):
    #show the original image
    w_name = window_name
    cv2.namedWindow(w_name, cv2.WINDOW_NORMAL) 
    
    # Using resizeWindow() 
    cv2.resizeWindow(w_name, 1920, 1080) 
    
    cv2.imshow(w_name, img) 
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def loadImagesBatch(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        images.append(img)
    return images
