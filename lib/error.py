import lib.util as util
import numpy as np
import cv2

#handles undistorted images
def computeProjectionError(homography, points: tuple, src_parameters = None, dst_parameters = None):
    """
    Computes the average L1_NORM between the target and the mapped source points with an homography  
    :param homography: the 3x3 homography between the points coordinates
    :param points: (src_points, target_points)distorted points tuple, src and target have to be the same shape 
    :param src_parameters: (only if computing with undistortion) the parameters of the source camera of the homography
    :param dst_parameters: (only if computing with undistortion) the parameters of the target camera of the homography
    :return: the reprojection error between the points sets
    """
    undflag = src_parameters is None or dst_parameters is None
    src_points, target_points = points
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
    
    #flag for handling the undistorted case
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
    #print(f'reprojection error: {error}')
    return error


def computeCameraReprojectionError(source_camera: str):
    """
    Given a camera it computes the average error and returns the reprojection error for each other camera
    :param source_camera: the index of the camera used ('1' or '2' or ...)
    :returns: (error_dict, average) tuple containing the dictionary of all errors for each camera and the average value {'undistorted','distorted'}
    """
    # loading all the intrinsic parameters
    parameters = util.load_json_parameters()
    parameters = util.parameters_to_numpy_array(parameters)
    # source camera parameters
    s_p = parameters[source_camera]
    mtx_src = s_p['mtx']
    dist_src = s_p['dist']
    new_mtx_src = s_p['new_mtx']
    # near and far cameras
    near, far = util.near_far_cameras(source_camera)

    camera_src = source_camera
    error_dict = {}
    average = {'undistorted': 0, 'distorted': 0}
    num = 9
    for camera_dst in near:
        d_p = parameters[camera_dst] 
        mtx2 = d_p['mtx']
        dist2 = d_p['dist']
        new_mtx2 = d_p['new_mtx']
        h12 = util.computeCamerasHomography(f'{camera_src}',f'{camera_dst}')
        h12_und = util.computeCamerasUndistortedHomography(f'{camera_src}',f'{camera_dst}', (mtx_src, mtx2), (dist_src, dist2), (new_mtx_src, new_mtx2))
        if h12 is None or h12_und is None:
            error_dict[camera_dst] = {'undistorted': -1, 'distorted': -1}
            num -= 1
            continue

        measures = util.LoadJSON('measures.json')

        camera_src_points = []
        camera_dst_points = []

        src_points = measures["image_points"][f"out{camera_src}"]
        dst_points = measures["image_points"][f"out{camera_dst}"]


        for key in src_points:
            temp = dst_points.get(key)
            if temp:
                camera_dst_points.append(temp)
                camera_src_points.append(src_points[key])

        camera_src_points = np.array(camera_src_points, np.float32)
        camera_dst_points = np.array(camera_dst_points, np.float32)

        points = (camera_src_points, camera_dst_points)
        src_parameters = (mtx_src, dist_src, new_mtx_src)
        dst_parameters = (mtx2, dist2, new_mtx2)

        und_err = computeProjectionError(h12_und, points, src_parameters, dst_parameters)
        std_err = computeProjectionError(h12, points)

        error_dict[camera_dst] = {'undistorted': und_err, 'distorted': std_err}
        average['undistorted'] += (und_err / num)
        average['distorted'] += (std_err / num) 

    camera_src = f"{near[-1]}"
    #print(camera_src)
    for camera_dst in far:
        d_p = parameters[camera_dst] 
        mtx2 = d_p['mtx']
        dist2 = d_p['dist']
        new_mtx2 = d_p['new_mtx']
        h12 = util.computeCamerasHomography(f'{camera_src}',f'{camera_dst}')
        h12_und = util.computeCamerasUndistortedHomography(f'{camera_src}',f'{camera_dst}', (mtx_src, mtx2), (dist_src, dist2), (new_mtx_src, new_mtx2))
        if h12 is None or h12_und is None:
            error_dict[camera_dst] = {'undistorted': -1, 'distorted': -1}
            num -= 1
            continue

        measures = util.LoadJSON('measures.json')

        camera_src_points = []
        camera_dst_points = []

        src_points = measures["image_points"][f"out{camera_src}"]
        dst_points = measures["image_points"][f"out{camera_dst}"]


        for key in src_points:
            temp = dst_points.get(key)
            if temp:
                camera_dst_points.append(temp)
                camera_src_points.append(src_points[key])

        camera_src_points = np.array(camera_src_points, np.float32)
        camera_dst_points = np.array(camera_dst_points, np.float32)

        points = (camera_src_points, camera_dst_points)
        src_parameters = (mtx_src, dist_src, new_mtx_src)
        dst_parameters = (mtx2, dist2, new_mtx2)

        und_err = computeProjectionError(h12_und, points, src_parameters, dst_parameters)
        std_err = computeProjectionError(h12, points)

        error_dict[camera_dst] = {'undistorted': und_err, 'distorted': std_err}
        average['undistorted'] += (und_err / num)
        average['distorted'] += (std_err / num) 
    
    return error_dict, average

def allCamerasReprojectionError():
    """
    :return: the dictioinary with all the average errors for each camera
    """
    cameras = ['1','2','3','4','5','6','7','8','12','13']

    err_dict = {}
    for camera in cameras:
        _, average = computeCameraReprojectionError(camera)
        err_dict[camera] = average

    return err_dict 