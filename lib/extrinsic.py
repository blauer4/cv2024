import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import yaml
import json


def import_coordinates():
    image_points_ret = {}
    world_points_ret = {}
    with open('measures.json', "r") as file:
        measures = json.load(file)
        world_points = measures["world_points"]
        image_points = measures["image_points"]
        # world_points = np.array(world_points, dtype=np.float32)
        for camera in image_points:
            camera_parameters_path = os.path.join(f"json/{camera}F/{camera.replace('out', '')}Fcorners_notc.json")
            camera_parameters = json.load(open(camera_parameters_path, "rb"))
            x, y, w, h = camera_parameters["roi"]
            mtx = np.array(camera_parameters["mtx"], dtype=np.float32)
            new_mtx = np.array(camera_parameters["new_mtx"], dtype=np.float32)
            distortion_coefficients = np.array(camera_parameters["dist"], dtype=np.float32)

            for point in image_points[camera]:
                if camera not in image_points_ret:
                    image_points_ret[camera] = []
                    world_points_ret[camera] = []

                # Undistort a point
                und_point = cv2.undistortPoints(np.array(image_points[camera][point], dtype=np.float32), mtx,
                                                distortion_coefficients, P=new_mtx)
                und_point = np.squeeze(und_point)
                if (und_point[0] - x) > w or (und_point[1] - y) > h:
                    print(f"Point {point} is outside the ROI in camera {camera}")
                else:
                    image_points_ret[camera].append(image_points[camera][point])
                    world_points_ret[camera].append(world_points[point])
            image_points_ret[camera] = np.array(image_points_ret[camera], dtype=np.float32)
            world_points_ret[camera] = np.array(world_points_ret[camera], dtype=np.float32)
            # print(image_points_ret[camera].shape, world_points_ret[camera])

    # print(world_points_ret, image_points_ret)

    return world_points_ret, image_points_ret


def pretty_print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:8.4f}" for val in row))


def plot_camera(extrinsic_matrices, all_camera_coordinates, size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    camera_position = []
    camera_direction = []
    # Plot camera direction obtained from extrinsic matrix
    direction_vector_size = 10

    # The camera positions are in order of the camera numbers
    for extrinsic_matrix in extrinsic_matrices:
        cam = extrinsic_matrix[:3, 3]
        camera_position.append(cam)
        direction = extrinsic_matrix[:3, :3] @ np.array([0, 0, direction_vector_size]) + cam
        camera_direction.append(direction)
        ax.plot([cam[0], direction[0]], [cam[1], direction[1]], [cam[2], direction[2]], c="g")
    # print(f'Calculated: {camera_position}\t{all_camera_coordinates}')
    # Plot camera location obtained from extrinsic matrix
    ax.scatter(
        [c[0] for c in camera_position],
        [c[1] for c in camera_position],
        [c[2] for c in camera_position],
        c="r",
        marker="o",
        label="Camera"
    )

    # Plot other camera positions
    ax.scatter(
        [coordinates[0] for coordinates in all_camera_coordinates.values()],
        [coordinates[1] for coordinates in all_camera_coordinates.values()],
        [coordinates[2] for coordinates in all_camera_coordinates.values()],
        c="c",
        marker="o",
        label="Other Cameras",
    )

    for camera_number, coordinates in all_camera_coordinates.items():
        ax.text(coordinates[0], coordinates[1], coordinates[2], camera_number)

    # Plot volleyball court points
    volleyball_points = np.array(
        [
            [9.0, 4.5, 0.0],
            [3.0, 4.5, 0.0],
            [-3.0, 4.5, 0.0],
            [-9.0, 4.5, 0.0],
            [9.0, -4.5, 0.0],
            [3.0, -4.5, 0.0],
            [-3.0, -4.5, 0.0],
            [-9.0, -4.5, 0.0],
        ],
        dtype=np.float32,
    )
    ax.scatter(
        volleyball_points[:, 0], volleyball_points[:, 1], volleyball_points[:, 2], c="b", marker="o", label="Points"
    )

    ax.set_xlim([camera_position[0][0] - size, camera_position[0][0] + size])
    ax.set_ylim([camera_position[0][1] - size, camera_position[0][1] + size])
    ax.set_zlim([camera_position[0][2] - size, camera_position[0][2] + size])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Position and Points")
    ax.legend()

    plt.show()


def main(arg):
    # Baseline camera coordinates
    all_camera_coordinates = {'camera_1': [14.5, 17.7, 6.2], 'camera_2': [0.0, 17.7, 6.2],
                              'camera_3': [22.0, 10.0, 6.6], 'camera_4': [-14.5, 17.7, 6.2],
                              'camera_5': [22.0, -10.2, 5.8], 'camera_6': [0.0, -10.0, 6.3],
                              'camera_7': [-25.0, 0.0, 6.4], 'camera_8': [-22.0, -10.0, 6.3],
                              'camera_12': [-22.4, 10.0, 6.9], 'camera_13': [22.0, 0.0, 7.0]}

    for x in all_camera_coordinates:
        all_camera_coordinates[x] = np.array(all_camera_coordinates[x], dtype=np.float32)

    world_points, image_points = import_coordinates()
    extrinsic_matrices = []
    for camera in image_points:

        camera_parameters_path = os.path.join(f"json/{camera}F/{camera.replace('out', '')}Fcorners_notc.json")

        if not os.path.exists(camera_parameters_path) or image_points[camera].size == 0:
            print(f"JSON file does not exist for {camera} or image_points")
            continue

        camera_parameters = json.load(open(camera_parameters_path, "rb"))
        # Using the new camera matrix because it is more accurate
        camera_matrix = np.array(camera_parameters["mtx"], dtype=np.float32)
        distortion_coefficients = np.array(camera_parameters['dist'], dtype=np.float32)

        # Solve PnP to calculate the extrinsic matrix
        success, rotation_vector, translation_vector = cv2.solvePnP(world_points[camera], image_points[camera],
                                                                    camera_matrix, distortion_coefficients)

        if not success:
            print("Failed to solve PnP", file=sys.stderr)
            sys.exit(1)

        # Convert the rotation vector to a rotation matrix using Rodrigues
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # THIS IS IMPORTANT
        # The output from solvePnP is the rotation vector and the translation vector
        # of the world coordinate system with respect to the camera coordinate system
        # We want the inverse of this transformation, so we invert the rotation matrix
        # to get the position and rotation of the camera with respect to the world
        # Note: the rotation_matrix can be inverted by transposing it, since it is orthonormal
        # I don't know if it gives some performance improvements, but for clarity I chose to use
        # the np.linalg.inv function
        inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
        inverse_translation_vector = -np.dot(inverse_rotation_matrix, translation_vector)

        extrinsic_matrix = np.hstack((inverse_rotation_matrix, inverse_translation_vector))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

        extrinsic_matrices.append(extrinsic_matrix)

        with open(f"json/{camera}F/{camera}_extrinsic_matrix.json", "w") as file:
            json.dump({
                'ext_mtx': extrinsic_matrix.tolist(),
                'rvec': rotation_vector.tolist(),
                'tvec': translation_vector.tolist(),
            }, file)

        # print(f"Camera {camera} extrinsic matrix:")
        # pretty_print_matrix(extrinsic_matrix)

    size = arg.size if arg.size else 10
    plot_camera(extrinsic_matrices, all_camera_coordinates, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show the extrinsic matrix of a camera")

    parser.add_argument("-s", "--size", type=int, help="The size of the plot")

    args = parser.parse_args()

    main(args)
