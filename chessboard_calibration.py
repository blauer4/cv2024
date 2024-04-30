import numpy as np
import cv2
import os
import re
import datetime as dt

all_chessboard_sizes = [(5, 7), (5, 7), (5, 7), (5, 7), (6, 9), (6, 9), (5, 7), (6, 9), (6, 9), (0, 0), (6, 9), (5, 7),
                        (5, 7)]

abs_dir_path = "scacchiere/"
camera_number = "8F"
frame_count = 0
# Set the frame skip interval 
frame_skip = 30
# Open the video file
video_capture = cv2.VideoCapture(f'{abs_dir_path}out{camera_number}.mp4')

# Print the number of frames in the video
frames_number = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Number of frames in the video: {frames_number}")

chessboard_size = all_chessboard_sizes[int(re.search("\d*", camera_number)[0]) - 1]

CHESS_WIDTH = chessboard_size[0]
CHESS_HEIGHT = chessboard_size[1]
VIDEO_NAME = 'out' + camera_number

# prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...(6,5,0)
objp = np.zeros((CHESS_WIDTH * CHESS_HEIGHT, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESS_WIDTH, 0:CHESS_HEIGHT].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

##Do we need it?
# Termination criteria for the iterative algorithm
# termination_criteria = (cv2.TERM_CRITERIA_EPS +
#                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Print the number of frames in the video
# numberOf_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
# print("Number of frames in the video: ", numberOf_frame)

# Create a directory to save the screen captures named as the video file
output_dir = 'samples/' + VIDEO_NAME
os.makedirs(output_dir, exist_ok=True)
print("Saving frames to ", output_dir)
chessboard_centers = np.array([[]])

# Remove all files in the directory
for file in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, file))

# Dictionary to keep track of the number of frames saved for each quadrant
quadrant_frame_count = {0: 0, 1: 0, 2: 0, 3: 0}
cosa = 0

now = dt.datetime.now()
timestamp = now.time()
print(timestamp)
while True:
    # Read a frame from the video
    ret, img = video_capture.read()
    if not ret:
        break  # Break the loop if we've reached the end of the video

    # Skip frames based on the frame_skip value
    # if frame_count % frame_skip == 0:
    # if frame_count in [200, 202, 230]:
    if frame_count % frame_skip == 0:
        print(frame_count)
        # Resize the frame to the new_resolution
        real_img = img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found_corners, corners = cv2.findChessboardCorners(gray, (CHESS_WIDTH, CHESS_HEIGHT), None)
        # print("Frame ", frame_count, ":", found_corners)

        # If found, add object points, image points, and save frames for each quadrant
        if found_corners:
            # centroid = (sum(corners[:, 0, 0]) / len(corners), sum(corners[:, 0, 1]) / len(corners))
            new_center = corners.mean(axis=0)
            print(corners.mean(axis=0))

            # print(f"Found corners: {found_corners}\t New center: {new_center}")
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("img", 1280, 720)

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
                    break
            if not skip:
                frame_filename = os.path.join(output_dir, f"chessboard{cosa}.jpg")
                cosa += 1
                chessboard_centers = np.append(chessboard_centers, new_center, axis=1)
                cv2.imwrite(frame_filename, img)
    if cosa > 20:
        break
    frame_count += 1
# print(chessboard_centers, cosa)
print(timestamp.microsecond - now.time().microsecond)
video_capture.release()
cv2.destroyAllWindows()
print("Camera", camera_number, "done!")

