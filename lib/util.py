import numpy as np
import json
import datetime
import cv2
import os


###
# METHODS AND CLASSES FOR GENERAL UTILITY
###

class Flag:
    f = True

    def __init__(self, value: bool):
        self.f = value

    def change(self):
        self.f = not self.f

def seeCamerasMapping(img, window_name, homographies):
    Hsrc, Hdst = homographies
    window_name_src, window_name_dst = window_name
    img_src, img_dst = img
    f = True

    if Hdst is None:
        f = False

    def mouseCallback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(f"event:{event}\nx:{x}\ny:{y}\nevent:{flags}")
            src_point = np.array([x,y,1])
            print(src_point)
            if f:
                dst_point = np.linalg.inv(Hdst) @ Hsrc @ src_point.T
            else:
                dst_point = Hsrc @ src_point.T
            print(dst_point)
            dst_point = dst_point.T
            print(dst_point)
            dst_point = np.array((dst_point/ dst_point[2]), dtype= np.int32) 
            print(f"{src_point} = {dst_point}")
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_src, f'[{x},{y}]', (x, y), font, 1, (0, 0, 0), 3)
            cv2.putText(img_dst, f'[{dst_point[0]},{dst_point[1]}]', (dst_point[0],dst_point[1]), font, 1, (0, 0, 0), 3)
            
            cv2.circle(img_src, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(img_dst, (dst_point[0],dst_point[1]), 5, (0, 255, 0), -1)
            cv2.imshow(window_name_src, img_src)
            cv2.imshow(window_name_dst, img_dst)


    def callbackButton(state, userdata):
        w_name, flag = userdata
        if flag.f:
            cv2.setMouseCallback(w_name, mouseCallback)
        else:
            cv2.setMouseCallback(w_name, lambda *args: None)
        flag.change()


    
    # show the original image
    cv2.namedWindow(window_name_src, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_dst, cv2.WINDOW_NORMAL)

    flag = Flag(True)

    cv2.createButton('select_pixel', callbackButton, (window_name_src, flag))

    # Using resizeWindow() 
    cv2.resizeWindow(window_name_src, 1920, 1080)
    cv2.resizeWindow(window_name_dst, 1920, 1080)
    cv2.imshow(window_name_src, img_src)
    cv2.imshow(window_name_dst, img_dst)

    cv2.waitKey(0)
    cv2.destroyWindow(window_name_src)
    cv2.destroyWindow(window_name_dst)
#insert the camera number of what you want to compute the homography
#uses measures files in the same directory of the calling file
def computeUndistortedHomography(camera:str, flags,save= True):

    measures = LoadJSON('measures.json')
    camera1_params = LoadJSON(f'json/out{camera}F/{camera}Fcorners_notc.json')
    #preparing for undistort the points
    mtx1 = np.array(camera1_params['mtx'])
    new_mtx1 = np.array(camera1_params['new_mtx'])
    dist1 = np.array(camera1_params['dist'])

    camera1_img = []
    camera1_world = []

    img_points = measures["image_points"][f"out{camera}"]
    world_points = measures["world_points"]

    for key in img_points:
        temp = world_points[key]
        camera1_world.append(temp)
        camera1_img.append(img_points[key])
    camera1_world = np.array(camera1_world)
    camera1_world = camera1_world[:,:2]
    camera1_img = np.array(camera1_img, np.float32)

    und_points = cv2.undistortPoints(camera1_img, mtx1, dist1, P=new_mtx1)
    Hom, mask = cv2.findHomography(und_points, camera1_world, method= flags)
    print(Hom)
    if save:
        saveToJSONstr({"H": Hom.tolist()},f'homography{camera}')
    return Hom

def computeCamerasHomography(src_camera, dst_camera,flags = 0,save= False):

    measures = LoadJSON('measures.json')

    camera_src = []
    camera_dst = []

    src_points = measures["image_points"][f"out{src_camera}"]
    dst_points = measures["image_points"][f"out{dst_camera}"]

    for key in src_points:
        temp = dst_points.get(key)
        if temp:
            camera_dst.append(temp)
            camera_src.append(src_points[key])
    camera_dst = np.array(camera_dst, np.float32)
    camera_dst = camera_dst[:,:2]
    camera_src = np.array(camera_src, np.float32)

    Hom, mask = cv2.findHomography(camera_src, camera_dst, method= flags)
    print(Hom)
    if save:
        saveToJSONstr({"H": Hom.tolist()},f'homography{src_camera}{dst_camera}')
    return Hom

#ugly version but works
def seeHomographyMapping(img, window_name, homography):
    def mouseCallback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(f"event:{event}\nx:{x}\ny:{y}\nevent:{flags}")
            img_point = np.array([x,y,1])
            real_point = homography @ img_point.T
            real_point = real_point/ real_point[2]
            print(real_point)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f'[{x},{y}], [{real_point}]', (x, y), font, 1, (0, 0, 0), 3)
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, img)


    def callbackButton(state, userdata):
        w_name, flag = userdata
        if flag.f:
            cv2.setMouseCallback(w_name, mouseCallback)
        else:
            cv2.setMouseCallback(w_name, lambda *args: None)
        flag.change()

    #to be changed
    def showImage(img, window_name: str):
        """
        Takes an image and show it in a fixed size window, press a button to close it in the end
            - img: An image to be shown
            - window_name: the name of the window where displaying the image
        """
        # show the original image
        w_name = window_name
        cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)

        flag = Flag(True)

        cv2.createButton('select_pixel', callbackButton, (w_name, flag))

        # Using resizeWindow() 
        cv2.resizeWindow(w_name, 1920, 1080)
        cv2.imshow(w_name, img)
        cv2.waitKey(0)
        # while cv2.waitKey(33) != ord('a'):
        #     cv2.imshow(w_name, img)
        cv2.destroyWindow(window_name)

    showImage(img,window_name)

###
# METHODS FOR MANIPULATING IMAGES
###

def showImage(img, window_name: str):
    """
    Takes an image and show it in a fixed size window, press a button to close it in the end
        - img: An image to be shown
        - window_name: the name of the window where displaying the image
    """
    # show the original image
    w_name = window_name
    cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)

    # Using resizeWindow() 
    cv2.resizeWindow(w_name, 1920, 1080)

    cv2.imshow(w_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def getMilliSeconds(dat: datetime):
    """
    Transform a datetime value in Milliseconds
        - dat: a datetime value
    """
    t = dat.time()
    return (t.hour * 60 * 60 * 1000) + (t.minute * 60 * 1000) + (t.second * 1000) + (t.microsecond / 1000)


###
# JSON MANIPULATION METHODS
###
def saveToJSON(dict: dict, camera_number: int):
    with open(f"{camera_number}corners.json", "a+") as outfile:
        json.dump(dict, outfile)
        return True
    return False

def saveToJSONstr(dict: dict, name: str):
    with open(f"{name}.json", "a+") as outfile:
        json.dump(dict, outfile)
        return True
    return False

def LoadJSON(filepath) -> dict:
    with open(filepath, "r") as outfile:
        return json.load(outfile)


###
# METHODS FOR LOADING IMAGES FROM DIRECTORY
###

def splitAndKeepNumber(li: str, d: list):
    """
    Returns the number present in "li" element and append it to d as [number, li].
        - li: a string in the form "img_name00.jpg"
        - d: the list where you want to store the mapping between the li string and it's number
    """
    numbers = np.arange(0, 10)
    numbers = list(map(str, numbers))
    before = ''
    for num in li:
        if num in numbers:
            temp = (li.split(before))[1]
            number = (temp.split('.'))[0]
            d.append([number, li])
            return number
        before = num


def NumericalSort(l: list) -> list:
    """
    Method to order the frame images,
      it accept a l list of names like "frame_name00.jpg" and order them only considering the first numerical part
        - l list of string values, each value should be in the form of a string value followed by a number and .jpg,
          otherwise the method will not operate correctly
    """
    temp = l.copy()
    d = []
    for i in temp:
        splitAndKeepNumber(i, d)
    d.sort(key=lambda x: int(x[0]))
    return d


def loadImagesBatch(folder, sorting=False):
    """
    Method that returns the list of images present in a directory.
        - folder: the relative path to the folder where are located the images
        - sorting: if True, sort the images in ascending order, filenames should be in "img_name00.jpg" format to work 
    """
    images = []
    l = os.listdir(folder)
    if sorting:
        l = NumericalSort(l)
        l = list(map(lambda x: x[1], l))
    print(f'list: {l}')
    for filename in l:
        img = cv2.imread(os.path.join(folder, filename))
        images.append(img)
    return images


###
# METHODS USED IN DEVELOPEMENT PHASE
###

def create2DArray(start_val, end_val):
    chess_2d_array = []
    val = np.arange(start_val, end_val)
    for a in val:
        for b in val:
            chess_2d_array.append([a, b])
    return chess_2d_array
