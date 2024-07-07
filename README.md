# cv2024

Project for the Computer Vision course of 2024

## Project Structure

The project is divided into four main files:

### 1. chessboard_calibration

This program computes the intrinsic parameters of a camera. It requires the following arguments:

- `camera_number`: The number of the camera you want to calibrate.
- `VIDEO_NAME`: The name of the video from which you want to extract the calibration frames.
- `frame_skip`: The number of frames to skip when analyzing the video. This should be small enough to avoid missing important frames.
- `batch_size`: The number of frames to process before performing a significant frame skip. A value between 2-4 is recommended to choose how many frames you want to extract.

The program saves the parameters in a JSON file structured as follows:

```
{
  "ret": reprojection error from the calibrateCamera output,
  "mtx": intrinsic parameter matrix
  "new_mtx": optimal intrinsic parameter matrix, obtained through getNewOptimalCameraMatrix
  "dist": distortion coefficients
  "roi": region of interest
  "error": reprojection error computed by us of mtx
  "new_error": reprojection error computed by us of new_mtx

}
```
Saves also all the images used for the calibration that are extracted by "findChessboardCorners"
### 2 extrinsic_calculation:
This program computes the extrinsic values of ... and shows the relative cameras position comparing it with the real world coordinates of the cameras
It needs the json of the intrinsic parameters computed in the first point.

The program saves the parameters in a JSON file structured as follows:
```
{
    "ext_mtx": the 4x4 matrix containing the extrinsic parameters of the camera
    "rvec": the rotation vector of the camera
    "tvec": the translation vector of the camera
 }

```
### 3 compute_homography:
Computes the homography from the given camera to all the others through the undistorted points.
It doesn't require arguments, but is necessary to have all the camera matrices and all the points corrispondeces.

The points corrispondences are located in `measures.json` 

The program saves the homographies in a JSON file structured as follows:
for camera '1' for example
```
{
    '2' : homography matrix from camera 1 to camera 2,
    '3' : homography matrix from camera 1 to camera 3
    ...    
}
```
and it will be named `homography1.json`

### 4 point_cameras: 
This program computes and shows the projection of a selected point to all the other cameras, through the homographies of the undistorted points.
It needs the json containing the intrinsic parameters of each camera and also the json containing all the homography matrices for each camera.
It requires the following arguments:

 - `camera_number`: The camera number where is possible to select the points to project. Passed as argument of the program by command line.

 By pressing the `display properties window` will appear a `select_pixel` button, which activate/deactivate the point selection on the source camera.
 Then just pressing any key will close the program.

## Additional Directories

The project also includes several other directories, each serving a specific purpose:

### `lib`

This directory contains the main and utility methods, including helper functions and classes that support the core calibration processes.

### `results`

This directory stores the output of the calibration processes, including undistorted images and other generated data.

### `test_images`

This directory holds images used to showcase the project's properties. It includes sample images from each camera, utilized for testing and demonstration purposes.

