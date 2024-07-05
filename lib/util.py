import numpy as np
import json
import datetime
import cv2
import os
import json


###
# METHODS AND CLASSES FOR GENERAL UTILITY
###

class Flag:
    f = True

    def __init__(self, value: bool):
        self.f = value

    def change(self):
        self.f = not self.f


def seeCamerasMapping(img, window_name, homographies, undistort=None):
    Hsrc, Hdst = homographies
    window_name_src, window_name_dst = window_name
    img_src, img_dst = img

    # undistort parameters

    mtx_src = None
    new_mtx_src = None
    dist_src = None
    mtx_dst = None
    dist_dst = None
    new_mtx_dst = None
    # flag for case handling
    homoflag = True
    undflag = True

    if Hdst is None:
        homoflag = False

    if undistort is None:
        undflag = False
    else:
        mtx_src, dist_src, new_mtx_src, mtx_dst, dist_dst, new_mtx_dst = undistort

    def mouseCallback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(f"event:{event}\nx:{x}\ny:{y}\nevent:{flags}")
            src_point = np.array([x, y], np.float32)
            dst_point = None
            print(f'src: {src_point}')
            if undflag:
                src_point = src_point[:2]
                src_point = cv2.undistortPoints(src_point, mtx_src, dist_src, P=new_mtx_src)
                src_point = np.append(src_point, [1])

            if homoflag:
                dst_point = np.linalg.inv(Hdst) @ Hsrc @ src_point.T
            else:
                dst_point = Hsrc @ src_point.T
            # Transpose it
            dst_point = dst_point.T
            # Normalize it
            dst_point = np.array((dst_point / dst_point[2]), dtype=np.float32)
            # Project back the point
            und_dst = cv2.undistortPoints(dst_point[:2], new_mtx_dst, np.zeros((1, 5), dtype=np.float32))
            dst_point = cv2.convertPointsToHomogeneous(und_dst)
            output = cv2.projectPoints(dst_point, np.zeros((1, 3), dtype=np.float32),
                                       np.zeros((1, 3), dtype=np.float32), mtx_dst, dist_dst, und_dst)

            output = output[0].flatten()
            x2 = int(output[0])
            y2 = int(output[1])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_src, f'[{x},{y}]', (x, y), font, 1, (0, 0, 0), 3)
            cv2.putText(img_dst, f'[{x2},{y2}]', (x2, y2), font, 1, (0, 0, 0), 3)

            cv2.circle(img_src, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(img_dst, (x2, y2), 5, (0, 255, 0), -1)

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


# insert the camera number of what you want to compute the homography
# uses measures files in the same directory of the calling file
def computeUndistortWorldHomography(camera: str, flags):
    """
    Compute the homography between the world points and the image points of a camera
    :param camera: The number of the camera to which compute the homography map
    :param flags: Flags for the opencv homography function
    :param save: If I want to save the homography map to file
    :return:
    """
    measures = LoadJSON('measures.json')
    camera1_params = LoadJSON(f'json/out{camera}F/{camera}Fcorners_notc.json')
    # preparing for undistort the points
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
    camera1_world = camera1_world[:, :2]
    camera1_img = np.array(camera1_img, np.float32)

    und_points = cv2.undistortPoints(camera1_img, mtx1, dist1, P=new_mtx1)
    hom, mask = cv2.findHomography(und_points, camera1_world, method=flags)
    return hom


# between the two distorted
def computeCamerasHomography(src_camera, dst_camera, flags=0):
    """
    Compute the homography between two cameras with distorted coordinates
    :param src_camera: The original camera from which we map the point
    :param dst_camera: The destination cam of the homography map
    :param flags: Flags for the opencv homography function
    :param save: If I want to save the homography map to file
    :return:
    """
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
    camera_dst = camera_dst[:, :2]
    camera_src = np.array(camera_src, np.float32)

    hom, mask = cv2.findHomography(camera_src, camera_dst, method=flags)
    return hom


# between the two rectified homography(better)
def computeCamerasUndistortedHomography(src_camera: str, dst_camera: str, mtx: tuple, dist: tuple, new_mtx: tuple,
                                        flags=0):
    """
    Compute the homography between two cameras with undistorted coordinates
    :param src_camera: The source camera from which we select a point
    :param dst_camera: The destination camera to which we map the point
    :param mtx: Tuple containing the camera matrix related to the src and dst cameras
    :param dist: Tuple containing the distortion vectors related to the src and dst cameras
    :param new_mtx: Tuple containing the undistorted camera matrix related to the src and dst cameras
    :param flags: Flags for the opencv homography function
    :return:
    """
    mtx_src, mtx_dst = mtx
    dist_src, dist_dst = dist
    new_mtx_src, new_mtx_dst = new_mtx
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

    camera_src = np.array(camera_src, np.float32)
    camera_dst = np.array(camera_dst, np.float32)

    if camera_dst.shape[0] < 4 or camera_src.shape[0] < 4:
        print("Too few points to compute the homography map between cam")
        hom = None
    else:
        camera_src = cv2.undistortPoints(camera_src, mtx_src, dist_src, P=new_mtx_src)
        camera_dst = cv2.undistortPoints(camera_dst, mtx_dst, dist_dst, P=new_mtx_dst)
        hom, _ = cv2.findHomography(camera_src, camera_dst, method=flags)
    return hom


def seeWorldHomographyMapping(img, window_name, homography):
    def mouseCallback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            img_point = np.array([x, y, 1])
            real_point = homography @ img_point.T
            real_point = real_point / real_point[2]
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

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    flag = Flag(True)

    cv2.createButton('select_pixel', callbackButton, (window_name, flag))

    cv2.resizeWindow(window_name, 1920, 1080)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def computeCameraProjectionError(homography, points: tuple, src_parameters = None, dst_parameters = None):

    src_points, target_points = points

    #flag for handling the undistorted case
    undflag = src_parameters is None or dst_parameters is None

    mtx_src = None
    dist_src = None
    new_mtx_src = None
    mtx_dst = None
    dist_dst = None
    new_mtx_dst = None

    if not undflag:
        mtx_src, dist_src , new_mtx_src = src_parameters
        mtx_dst, dist_dst , new_mtx_dst = dst_parameters
    h = homography
    points_number = src_points.shape[0]


    und_src_points = []

    if points_number != target_points.shape[0]:
        print("points numbers in source and target have to be the same")
        return -1
    if not undflag:
        und_src_points = cv2.undistortPoints(src_points, mtx_src, dist_src, P= new_mtx_src)
    else:
        und_src_points = src_points

    und_src_points = cv2.convertPointsToHomogeneous(und_src_points)
    und_src_points = np.squeeze(und_src_points,axis= 1)

    #compute the homography for each point
    dst_points = []
    for und_point in und_src_points:
        dst_point = h @ und_point.T
        #transpose it
        dst_point = dst_point.T
        #normalize it
        dst_point = dst_point/ dst_point[2]
        #print(f'dst_point:{dst_point}')
        dst_points.append(dst_point[:2])

    #print(f'dst_points:{dst_points}')
    dst_points = np.array(dst_points, dtype= np.float32)
    #print(f'dst:{dst_points}')

    #Project back the point
    if not undflag:
        dist_zero = np.zeros((1,5), np.float32)
        und_dst_points = cv2.undistortPoints(dst_points, new_mtx_dst, dist_zero)
        #print(f'dst after undistortion: {und_dst_points}')

        dst_points_hmgn = cv2.convertPointsToHomogeneous(und_dst_points)
        #print(f'dst sfter homogeneous: {dst_point}')

        tvec = np.zeros((1,3), dtype= np.float32)
        rvec = tvec
        output = cv2.projectPoints(dst_points_hmgn, rvec , tvec, mtx_dst, dist_dst)
        output = np.squeeze(output[0])
        #print(f'target:{target_points}\noutput:{output}')
    else:
        output = dst_points
    error = cv2.norm(output,target_points,normType=cv2.NORM_L1) / int(points_number)
    print(f'reprojection error: {error}')
    return error


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


def saveToJSONstr(dict: dict, name: str):
    with open(f"{name}.json", "a+") as outfile:
        json.dump(dict, outfile)
        return True


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
