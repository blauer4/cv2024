#point projection corrispondences
import lib.util as util
import cv2
import numpy as np

def seeCamerasMapping(img, window_name, homographies, undistort = None):
    Hsrc, Hdst = homographies
    window_name_src, window_name_dst = window_name
    img_src, img_dst = img
    
    #undistort parameters
    
    mtx_src = None
    new_mtx_src = None
    dist_src = None
    mtx_dst = None
    dist_dst = None
    new_mtx_dst = None
    #flag for case handling
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
            src_point = np.array([x,y], np.float32)
            dst_point = None
            print(f'src: {src_point}')
            if undflag:
                src_point = src_point[:2]
                src_point = cv2.undistortPoints(src_point, mtx_src, dist_src, P= new_mtx_src)
                src_point = np.append(src_point, [1])

            if homoflag:
                dst_point = np.linalg.inv(Hdst) @ Hsrc @ src_point.T
            else:
                dst_point = Hsrc @ src_point.T
            #transpose it
            dst_point = dst_point.T
            #normalize it 
            dst_point = np.array((dst_point/ dst_point[2]), dtype= np.float32) 
            #print(f"{src_point} = {dst_point}")
            print(f'dst:{dst_point}')
            #Project back the point
            und_dst = cv2.undistortPoints(dst_point[:2],new_mtx_dst, np.zeros((1,5), np.float32))
            print(f'dst after undistortion: {und_dst}')
            dst_point = cv2.convertPointsToHomogeneous(und_dst)
            print(f'dst sfter homogeneous: {dst_point}')
            output = cv2.projectPoints(dst_point, np.zeros((1,3), dtype= np.float32), np.zeros((1,3), dtype= np.float32), mtx_dst, dist_dst)
            print(f'after project: {output}')
            output = output[0].flatten()
            x2 = int(output[0])
            y2 = int(output[1])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_src, f'[{x},{y}]', (x, y), font, 1, (0, 0, 0), 3)
            cv2.putText(img_dst, f'[{x2},{y2}]', (x2,y2), font, 1, (0, 0, 0), 3)
            
            cv2.circle(img_src, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(img_dst, (x2,y2), 5, (0, 255, 0), -1)
            
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

    flag = util.Flag(True)

    cv2.createButton('select_pixel', callbackButton, (window_name_src, flag))

    # Using resizeWindow() 
    cv2.resizeWindow(window_name_src, 1920, 1080)
    cv2.resizeWindow(window_name_dst, 1920, 1080)
    cv2.imshow(window_name_src, img_src)
    cv2.imshow(window_name_dst, img_dst)

    cv2.waitKey(0)
    cv2.destroyWindow(window_name_src)
    cv2.destroyWindow(window_name_dst)

def computeCamerasUndistortedHomography(src_camera: str, dst_camera: str, mtx, dist, new_mtx, camera_name,flags = 0,save= False):
    mtx_src , mtx_dst = mtx
    dist_src, dist_dst = dist
    new_mtx_src, new_mtx_dst = new_mtx
    measures = util.LoadJSON('measures.json')

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
    camera_src = cv2.undistortPoints(camera_src, mtx_src, dist_src, P= new_mtx_src)
    camera_dst = np.array(camera_dst, np.float32)
    camera_dst = cv2.undistortPoints(camera_dst, mtx_dst, dist_dst, P= new_mtx_dst)

    Hom, mask = cv2.findHomography(camera_src, camera_dst, method= flags)
    print(Hom)
    if save:
        util.saveToJSONstr({"H": Hom.tolist()},f'homography{camera_name}')
    return Hom

#load the image you want to visualize
i1 = cv2.imread('camera1.png')
i2 = cv2.imread('camera3.png')
#load the intrinsic parameters
camera1_params = util.LoadJSON('json/out1F/1Fcorners_notc.json')
camera2_params = util.LoadJSON('json/out3F/3Fcorners_notc.json')

mtx1 = np.array(camera1_params['mtx'], dtype=np.float32)
new_mtx1 = np.array(camera1_params['new_mtx'], dtype=np.float32)
dist1 = np.array(camera1_params['dist'], dtype=np.float32)

mtx2 = np.array(camera2_params['mtx'], dtype=np.float32)
new_mtx2 = np.array(camera2_params['new_mtx'], dtype=np.float32)
dist2 = np.array(camera2_params['dist'], dtype=np.float32)

##From camera to undistorted to undistorted to camera
i1_und = cv2.undistort(i1,mtx1,dist1,None,new_mtx1)
i2_und = cv2.undistort(i2,mtx2,dist2,None,new_mtx2)

h12_und = computeCamerasUndistortedHomography('1','3', (mtx1, mtx2), (dist1, dist2), (new_mtx1, new_mtx2), 'homography13')
undistort = (mtx1, dist1, new_mtx1, mtx2, dist2, new_mtx2)
seeCamerasMapping((i1,i2), ('camera1','camera2'), (h12_und, None), undistort)