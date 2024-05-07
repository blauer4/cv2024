import numpy as np
import json
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
def saveToJSON(dict, camera_number):
    with open(f"{camera_number}Fcorners.json", "a+") as outfile:
        json.dump(dict, outfile)
        return True
    return False

class Calibration:
    """
    Class for executing the chessboard calibration:
        - path: is the path to the video to analize
        - width: is the number of columns of the chessboard present in the "path" video
        - height: is the number of rows of the chessboard present in the "path" video
    """
    def __init__(self, path, width, height) -> None:
        self.video_capture = cv2.VideoCapture(filename=path)
        self.total_frame_number = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.chess_width = width
        self.chess_height = height
        
        self.frame_number = 0
        self.img_number = 0


    def extractCorners(self, img, output_dir, subpix_window = (5,5), subpix_tc = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 15, 0.01)):
        if hasattr(self, 'chessboard_centers'):
            self.chessboard_centers = np.array([[]])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found_corners, corners = cv2.findChessboardCorners(gray, (self.chess_width, self.chess_height), None, flags= cv2.CALIB_CB_FAST_CHECK)
        # print("Frame ", frame_count, ":", found_corners)

        # If found, add object points, image points, and save frames for each quadrant
        if found_corners:
            # centroid = (sum(corners[:, 0, 0]) / len(corners), sum(corners[:, 0, 1]) / len(corners))
            new_center = corners.mean(axis=0)
            print(corners.mean(axis=0))


            skip = False
            for center in chessboard_centers:
                if center.size == 0: continue
                # print(f"Center: {center[0]}, {center[1]}")
                # print((new_center[0][0] - center[0]) ** 2 + (new_center[0][1] - center[1]) ** 2 )
                # We check that the points are inside a box around the center of distance dx<100
                dx = abs(center[0] - new_center[0][0])
                dy = abs(center[1] - new_center[0][1])
                # if ((new_center[0][0] - center[0]) ** 2 + (new_center[0][1] - center[1]) ** 2) < (1000 ** 2):
                if dx < 100 or dy < 100:
                    print("skipped frame because center are too near")
                    skip = True
                    return (False, None)
            if not skip:
                print("saving images to directory")
                frame_filename = os.path.join(output_dir, f"chessboard{self.img_number}.jpg")
                self.img_number += 1
                
                chessboard_centers = np.append(chessboard_centers, new_center, axis=1)
                cv2.imwrite(frame_filename, img)
                #computing sub pixels for better results
                sub_corners = cv2.cornerSubPix(gray, corners, subpix_window, (-1,-1), subpix_tc)
                return (True, sub_corners)

        else:
            return (False, None)
                

    #aim to extract at max 50 images
    def fastCalibrationImageSearch(self, funct= extractCorners, output_dir = 'samples/', skip_step = 2):
        """
        Compute the search of images and chessboard corners of the video
            - funct: a method which implement a criteria for searching in a frame.
                - funct_ret:  is a boolean value which tells if something has been found
                - result: - the data to be saved in a dict form 
            - output_dir: the directory where saving the calbration frames
            - skip_step: how many frames skip in the accurate phase of search
        Returns:
            - the list of imgpoints
        """
        imgpoints = []
        big_skip = int(self.total_frame_number / 25) 
        frame_number = self.frame_number
        
        print(f"{frame_number}, {self.total_frame_number}")
        
        while frame_number < self.total_frame_number:
            
            ret, img = self.video_capture.read()
            if not ret:
                break

            small_count = 0
            #start with a small skip step until you don't found 2 good corrispondence
            while small_count < 2:
                if frame_number % skip_step == 0:
                    funct_ret, corners = funct(img, output_dir)
                    if funct_ret:
                        print(f'find one: {frame_number}')
                        small_count+=1
                        imgpoints.append(corners)
                ret, img = self.video_capture.read()
                if not ret:
                    break
                frame_number += 1 
            if not ret:
                break
            #then jump to the next big frame skip
            for i in range(big_skip):
                print(f'starting big skip of {big_skip}: {frame_number}')
                ret, img = self.video_capture.read()
                if not ret:
                    break
                frame_number += 1
                print(f'frame after skip: {frame_number}')
        print(f"video corners search ended, saving images to {output_dir}")
        self.imgpoints = imgpoints
        return imgpoints
