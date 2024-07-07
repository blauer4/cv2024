import lib.util as util
import cv2
import numpy as np
import argparse
import sys


def show_image_grid(imgs, window_name, width, height):
    # Create a grid of images
    grid = cv2.vconcat([cv2.hconcat([imgs[0], imgs[1], imgs[2]]), cv2.hconcat([imgs[3], imgs[4], imgs[5]]),
                        cv2.hconcat([imgs[6], imgs[7], imgs[8]])])
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, grid)


def compute_project_point(img, homography, point, source_parameters, dst_parameters):
    src_point = point
    mtx_src, dist_src, new_mtx_src = source_parameters
    mtx_dst, dist_dst, new_mtx_dst = dst_parameters
    Hsrc = homography
    # print(f'src: {src_point}')
    src_point = src_point[:2]
    src_point = cv2.undistortPoints(src_point, mtx_src, dist_src, P=new_mtx_src)
    src_point = np.append(src_point, [1])

    dst_point = Hsrc @ src_point.T
    # Transpose it
    dst_point = dst_point.T
    # Normalize it
    dst_point = np.array((dst_point / dst_point[2]), dtype=np.float32)
    # Project back the point
    und_dst = cv2.undistortPoints(dst_point[:2], new_mtx_dst, np.zeros((1, 5), dtype=np.float32))
    dst_point = cv2.convertPointsToHomogeneous(und_dst)
    output = cv2.projectPoints(dst_point, np.zeros((1, 3), dtype=np.float32),
                                       np.zeros((1, 3), dtype=np.float32), mtx_dst, dist_dst)
    
    output = output[0].flatten()
    x2 = int(output[0])
    y2 = int(output[1])
    # print(f"output: {[x2, y2]}")
    if x2 < 0 or y2 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        return img, [0, 0]
    else:
        cv2.putText(img, f'[{x2},{y2}]', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.circle(img, (x2, y2), 10, (0, 255, 0), -1)

    return img, [x2, y2]


def run_camera_mappings(source_camera: str):
    # loading all the images where project the points
    images = util.load_test_images()
    # loading all the intrinsic parameters
    parameters = util.load_json_parameters()
    parameters = util.parameters_to_numpy_array(parameters)
    # source camera parameters
    s_p = parameters[source_camera]
    source_parameters = (s_p['mtx'], s_p['dist'], s_p['new_mtx'])
    # near and far cameras
    near, far = util.near_far_cameras(source_camera)
    img_src = images[source_camera]

    def mouse_callback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            h = util.LoadJSON(f'json/out{source_camera}F/homography{source_camera}.json')
            homography_source = util.to_numpy_array(h)
            img_s = params[0]
            src_parameters = params[1]
            point = np.array([x, y], np.float32)
            x, y = int(point[0]), int(point[1])
            near_points = {}
            cv2.putText(img_s, f'[{x},{y}]', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.circle(img_s, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(main_window, img_s)

            all_points_images = []
            for camera in near:
                dst_p = parameters[camera]
                dst_parameters = (dst_p['mtx'], dst_p['dist'], dst_p['new_mtx'])
                img, projected_point = compute_project_point(images[camera], homography_source[camera], point,
                                                             src_parameters, dst_parameters)
                near_points[camera] = projected_point
                all_points_images.append(img)

            # now computing the furthest cameras
            second_camera = near[-1]
            # taking the projected point from before as a new source point
            point = np.array(near_points[second_camera], dtype=np.float32)
            #print(f'starting near point: {point}')
            # load all the homographies for computing the remaining cameras
            src_parameters = (parameters[second_camera]['mtx'], parameters[second_camera]['dist'],
                              parameters[second_camera]['new_mtx'])
            homography_second = util.LoadJSON(f'json/out{second_camera}F/homography{second_camera}.json')
            for camera in far:
                dst_p = parameters[camera]
                dst_parameters = (dst_p['mtx'], dst_p['dist'], dst_p['new_mtx'])
                img, p = compute_project_point(images[camera], homography_second[camera], point, src_parameters,
                                               dst_parameters)
                all_points_images.append(img)

            show_image_grid(all_points_images, 'All cameras', 1920, 1080)

    def callback_button(state, userdata):
        w_name, flag, img_s, src_param = userdata
        if flag.f:
            cv2.setMouseCallback(w_name, mouse_callback, (img_s, src_param))
        else:
            cv2.setMouseCallback(w_name, lambda *args: None)
        flag.change()

    flag = util.Flag(True)
    main_window = f'Source: camera {source_camera}'
    cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(main_window, 1280, 720)
    cv2.imshow(main_window, img_src)
    cv2.createButton('select_pixel', callback_button, (main_window, flag, img_src, source_parameters))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show a point from one camera to all the others")
    parser.add_argument("camera_number", type=int, help="The source camera number from which we project the point")
    args = parser.parse_args()

    if not args.camera_number:
        print("Please provide a camera number", file=sys.stderr)
        exit(1)

    run_camera_mappings(str(args.camera_number))
