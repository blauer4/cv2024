import cv2
import numpy as np
import os


class Calibration:
    """
    Class for executing the chessboard calibration:
        - path: is the path to the video to analyze
        - width: is the number of columns of the chessboard present in the "path" video
        - height: is the number of rows of the chessboard present in the "path" video
        - size: is the size of the video frames express as (width, height)
    """

    def __init__(self, path, width, height) -> None:
        # video info
        self.imgpoints = []
        self.chessboard_centers = np.array([[]])
        self.video_capture = cv2.VideoCapture(filename=path)
        self.size = (int(self.video_capture.get(3)), int(self.video_capture.get(4)))
        self.img_number = 0
        self.frame_number = 0
        self.total_frame_number = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # chessboard info
        self.chess_width = width
        self.chess_height = height

    def __str__(self):
        return f'total video frames:{self.total_frame_number}\nsize:{self.size}\nchessboard width:{self.chess_width}\nchess height:{self.chess_height}\n'

    def extractCorners(self, img, output_dir: str,
                       subpix_window=(10, 10),
                       subpix_tc=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 40, 0.01),
                       find_flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK):

        chessboard_centers = self.chessboard_centers
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found_corners, corners = cv2.findChessboardCorners(gray, (self.chess_width, self.chess_height), None,
                                                           flags=find_flags)
        # print("Frame ", frame_count, ":", found_corners)
        # If found, add object points, image points, and save frames for each quadrant
        if found_corners:
            if isNearBorder(corners, self.size[0], self.size[1], self.chess_width, self.chess_height):
                print("skipped frame because too near the image border")
                return False, None
            # centroid = (sum(corners[:, 0, 0]) / len(corners), sum(corners[:, 0, 1]) / len(corners))
            new_center = corners.mean(axis=0)
            # print(corners.mean(axis=0))

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
                    return False, None

            print("saving images to directory")
            frame_filename = os.path.join(output_dir, f"chessboard{self.img_number}.jpg")
            self.img_number += 1

            self.chessboard_centers = np.append(chessboard_centers, new_center, axis=1)
            cv2.imwrite(frame_filename, img)
            # computing sub pixels for better results
            sub_corners = cv2.cornerSubPix(gray, corners, subpix_window, (-1, -1), subpix_tc)
            return True, sub_corners

        else:
            return False, None

    def extractCornersNoSub(self, img, output_dir: str,
                            find_flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK):
        if not hasattr(self, 'chessboard_centers'):
            self.chessboard_centers = np.array([[]])

        chessboard_centers = self.chessboard_centers
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found_corners, corners = cv2.findChessboardCorners(gray, (self.chess_width, self.chess_height), None,
                                                           flags=find_flags)
        if isNearBorder(corners, self.size[0], self.size[1], self.chess_width, self.chess_height):
            print("skipped frame because too near the image border")
            return False, None
        # If found, add object points, image points, and save frames for each quadrant
        if found_corners:
            new_center = corners.mean(axis=0)
            print(corners.mean(axis=0))

            skip = False
            for center in chessboard_centers:
                if center.size == 0: continue
                # We check that the points are inside a box around the center of distance dx<100
                dx = abs(center[0] - new_center[0][0])
                dy = abs(center[1] - new_center[0][1])
                if dx < 100 or dy < 100:
                    print("skipped frame because center are too near")
                    skip = True
                    return False, None
            if not skip:
                print("saving images to directory")
                frame_filename = os.path.join(output_dir, f"chessboard{self.img_number}.jpg")
                self.img_number += 1
                chessboard_centers = np.append(chessboard_centers, new_center, axis=1)
                cv2.imwrite(frame_filename, img)

                return True, corners

        else:
            return False, None

    def fastImagesSearch(self, funct=None, output_dir='samples', skip_step=2, batch_size=3, release_video=True):
        """
        Compute the search of images and chessboard corners of the video, release the video if "release_video" is a true
            - funct: a method which implement a criteria for searching in a frame.
                - funct_ret:  is a boolean value which tells if something has been found
                - result: - the data to be saved in a dict form 
            - output_dir: the directory where saving the calibration frames
            - skip_step: how many frames skip in the accurate phase of search
            - batch_size: how many frames I want to save before doing a big skip
                -Try to not put this number too big, because it will basically scan all the frames
                with only the "skip_step"
            - release_video: if True, it will release the videoCapture
        Returns:
            - the list of imgpoints
        """
        if funct is None:
            funct = self.extractCornersNoSub

        imgpoints = []
        objpoints = []

        objp = np.zeros((self.chess_width * self.chess_height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chess_width, 0:self.chess_height].T.reshape(-1, 2)

        big_skip = int(self.total_frame_number / 25)
        frame_number = self.frame_number

        # print(f"{frame_number}, {self.total_frame_number}")
        count = 0
        while frame_number < self.total_frame_number:

            ret, img = self.video_capture.read()
            if not ret:
                break

            small_count = 0
            # start with a small skip step until you don't found 2 good corrispondence
            while small_count < batch_size:
                if frame_number % skip_step == 0:
                    funct_ret, corners = funct(img, output_dir)
                    count += 1
                    if funct_ret:
                        print(f'find one: {frame_number}')
                        small_count += 1
                        imgpoints.append(corners)
                        objpoints.append(objp)
                ret, img = self.video_capture.read()
                if not ret:
                    break
                frame_number += 1
            if not ret:
                break
            # then jump to the next big frame skip
            print(f'starting big skip of {big_skip}: {frame_number}')
            for i in range(big_skip):
                ret, img = self.video_capture.read()
                if not ret:
                    break
                frame_number += 1
            print(f'frame after skip: {frame_number}')
        print(f"video corners search ended, saving images to {output_dir}")
        self.imgpoints = imgpoints
        print(f"frames analyzed: {count}")
        if release_video:
            print("releasing video")
            self.video_capture.release()
        return imgpoints, objpoints


def computeReProjError(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """
    method for computing multiple reprojeciton errors.
    It returns the average value of the reprojection errors, different from the openCv calibrateCamera ret value(which is RMSE)
        - objpoints: list of object points in the 3D camera space
        - imgpoints: list of image points in the 2D camera plane
        - mtx: camera matrix (3x3)
        - dist: distorsion values of the camera
        - rvecs: list of rotation vectors
        - tvecs: list of translation vectors
    """
    mean_error = 0
    for i in range(np.shape(objpoints)[0]):
        # project the points
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # calculate the ecludian distance of all the point
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error / (np.shape(objpoints)[0])
    return mean_error


def computeMultiplePnP(objpoints, imgpoints, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE):
    tvecs = []
    rvecs = []
    for i in range(np.shape(objpoints)[0]):
        r, rvec, tvec = cv2.solvePnP(objpoints[i], imgpoints[i], mtx, dist, flags)
        rvecs.append(rvec)
        tvecs.append(tvec)
    return rvecs, tvecs


# Refine case of solvePnP
def computeMultipleRefinePnP(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    final_tvecs = []
    final_rvecs = []
    for i in range(np.shape(objpoints)[0]):
        # print(rvecs[i])
        rvec, tvec = cv2.solvePnPRefineLM(objpoints[i], imgpoints[i], mtx, dist, rvecs[i], tvecs[i])
        final_rvecs.append(rvec)
        final_tvecs.append(tvec)
    return final_rvecs, final_tvecs


def isNearBorder(corners, width, height, chess_width, chess_height):
    x = np.array([])
    y = np.array([])
    for corner in corners:
        x = np.append(x, corner[0, 0])
        y = np.append(y, corner[0, 1])
    xmx = x.max()
    ymx = y.max()
    xd = xmx - x.min()
    yd = ymx - y.min()
    # print(f"x:{x}")
    # print(f"y: {y}")
    threshold = (xd / chess_width) + (yd / chess_height)
    # print(f"threshold: {threshold}")
    check1 = x < threshold
    check2 = y < threshold
    check3 = (width - x) < threshold
    check4 = (height - y) < threshold
    if (check1.any()) or (check2.any()) or (check3.any()) or (check4.any()):
        return True
    return False
