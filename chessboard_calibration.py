import numpy as np
import cv2
import os
import re
import datetime as dt
import lib.util as util
import lib.Calibration as cal

all_chessboard_sizes = [(5, 7), (5, 7), (5, 7), (5, 7), (6, 9), (6, 9), (5, 7), (6, 9), (6, 9), (0, 0), (6, 9), (5, 7),
                        (5, 7)]

camera_number = "5F"
VIDEO_NAME = 'out' + camera_number
# Set the frame skip interval 
frame_skip = 10

chessboard_size = all_chessboard_sizes[int(re.search("\d*", camera_number)[0]) - 1]

CHESS_WIDTH = chessboard_size[0]
CHESS_HEIGHT = chessboard_size[1]

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Create a directory to save the screen captures named as the video file
output_dir = 'samples/' + VIDEO_NAME
os.makedirs(output_dir, exist_ok=True)
print("Saving frames to ", output_dir)
chessboard_centers = np.array([[]])

# Remove all files in the directory
for file in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, file))

relative_path = "videos/"
VIDEO_NAME = f'out{camera_number}.mp4'
video_path = f'{relative_path}{VIDEO_NAME}'

video = cal.Calibration(video_path, CHESS_WIDTH, CHESS_HEIGHT)
print(video)

now = dt.datetime.now()
timestamp = util.getMilliSeconds(now)

#compute the corners research
imgpoints, objpoints = video.fastImagesSearch(funct= video.extractCorners, output_dir= output_dir, skip_step=frame_skip)
print(f'number of corners detected: {len(imgpoints)}')
print(f'first imgpoints: {imgpoints[0]}')
now = dt.datetime.now()
t2 = util.getMilliSeconds(now)
print(f"number of seconds: {(t2-timestamp) / 1000}")

if imgpoints is []:
    print('nothing detected.. exiting')
    exit(0)

#imgpoints is an array of an array of the detected points on the image
camera_tc = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 0.005, 50)
print(f'tc: {camera_tc}')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, video.size, None, camera_tc)
#print(f"ret:{ret}\nmtx:{mtx}\ndist:{dist}\nrvecs:{rvecs}\ntvecs:{tvecs}")
print(video.size)
json_camera_matrix = {
        'ret' : ret,
        'mtx' : mtx.tolist(),
        'dist': dist.tolist(),
        'rvecs' : [rvecs[0].tolist(), rvecs[1].tolist()],
        'tvecs' : [tvecs[0].tolist(), tvecs[1].tolist()]
}
util.saveToJSON(json_camera_matrix, camera_number)

print("Camera", camera_number, "done!")

