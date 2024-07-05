import os
import lib.util as util
import cv2
import numpy as np


if __name__ == "__main__":
    camera_numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "12", "13"]
    all_hom = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {}, "6": {}, "7": {}, "8": {}, "12": {}, "13": {}}
    for camera_number in camera_numbers:
        camera_json = {}
        for second_camera in camera_numbers:
            if (second_camera == camera_number or second_camera in all_hom[camera_number].keys()
                    or camera_number in all_hom[second_camera].keys()):
                continue
            camera1_params = util.LoadJSON(f'json/out{camera_number}F/{camera_number}Fcorners_notc.json')
            camera2_params = util.LoadJSON(f'json/out{second_camera}F/{second_camera}Fcorners_notc.json')

            mtx1 = np.array(camera1_params['mtx'], dtype=np.float32)
            new_mtx1 = np.array(camera1_params['new_mtx'], dtype=np.float32)
            dist1 = np.array(camera1_params['dist'], dtype=np.float32)

            mtx2 = np.array(camera2_params['mtx'], dtype=np.float32)
            new_mtx2 = np.array(camera2_params['new_mtx'], dtype=np.float32)
            dist2 = np.array(camera2_params['dist'], dtype=np.float32)

            h = util.computeCamerasUndistortedHomography(camera_number, second_camera, (mtx1, mtx2),
                                                         (dist1, dist2), (new_mtx1, new_mtx2))
            if h is not None:
                all_hom[camera_number][second_camera] = h.tolist()
                all_hom[second_camera][camera_number] = h.tolist()

        util.saveToJSONstr(all_hom[camera_number], f'json/out{camera_number}F/homography{camera_number}')
