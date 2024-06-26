import argparse
import cv2 as cv
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
            for point in image_points[camera]:
                if camera not in image_points_ret:
                    image_points_ret[camera] = []
                    world_points_ret[camera] = []
                image_points_ret[camera].append(image_points[camera][point])
                world_points_ret[camera].append(world_points[point])
            image_points_ret[camera] = np.array(image_points_ret[camera], dtype=np.float32)
            world_points_ret[camera] = np.array(world_points_ret[camera], dtype=np.float32)
            # print(image_points_ret[camera].shape, world_points_ret[camera].shape)

    # print(world_points_ret, image_points_ret)

    return world_points_ret, image_points_ret


def pretty_print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:8.4f}" for val in row))


def plot_camera(extrinsic_matrix, all_camera_coordinates, size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # The camera positions are in order of the camera numbers

    camera_position = extrinsic_matrix[:3, 3]

    # Plot camera location obtained from extrinsic matrix
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c="r", marker="o", label="Camera")

    # Plot camera direction obtained from extrinsic matrix
    direction_vector_size = 10
    camera_direction = extrinsic_matrix[:3, :3] @ np.array([0, 0, direction_vector_size]) + camera_position
    ax.plot(
        [camera_position[0], camera_direction[0]],
        [camera_position[1], camera_direction[1]],
        [camera_position[2], camera_direction[2]],
        c="g",
        label="Camera Direction",
    )

    # Plot other camera positions
    ax.scatter(
        [coordinates[0] for coordinates in all_camera_coordinates.values()],
        [coordinates[1] for coordinates in all_camera_coordinates.values()],
        [coordinates[2] for coordinates in all_camera_coordinates.values()],
        c="y",
        marker="o",
        label="Other Cameras",
    )

    for camera_number, coordinates in all_camera_coordinates.items():
        ax.text(coordinates[0], coordinates[1], coordinates[2], str(camera_number))

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

    ax.set_xlim([camera_position[0] - size, camera_position[0] + size])
    ax.set_ylim([camera_position[1] - size, camera_position[1] + size])
    ax.set_zlim([camera_position[2] - size, camera_position[2] + size])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Position and Points")
    ax.legend()

    plt.show()


def main(arg):
    camera_number = arg.camera_number

    if arg.with_crop:
        with_crop = True
    elif arg.without_crop:
        with_crop = False
    else:
        print("Invalid arguments", file=sys.stderr)
        sys.exit(1)

    world_points, image_points = import_coordinates()

    camera_parameters_path = os.path.join(f"{camera_number}Fcorners.json")
    camera_parameters = json.load(open(camera_parameters_path, "rb"))

    if not os.path.exists(camera_parameters_path):
        print("Pickle file does not exist")
        sys.exit(1)

    all_camera_coordinates = {'camera_1': [14.5, 17.7, 6.2], 'camera_2': [0.0, 17.7, 6.2],
                              'camera_3': [22.0, 10.0, 6.6], 'camera_4': [-14.5, 17.7, 6.2],
                              'camera_5': [22.0, -10.2, 5.8], 'camera_6': [0.0, -10.0, 6.3],
                              'camera_7': [-25.0, 0.0, 6.4], 'camera_8': [-22.0, -10.0, 6.3],
                              'camera_9': [-14.5, -17.7, 6.2], 'camera_12': [-22.0, 10.0, 6.9],
                              'camera_13': [22.0, 0.0, 7.0]}

    for x in all_camera_coordinates:
        all_camera_coordinates[x] = np.array(all_camera_coordinates[x], dtype=np.float32)

    camera_matrix = np.array(camera_parameters["mtx"])
    # new_mtx does not exist in the pickle file if the image was undistorted with crop
    new_camera_matrix = None if with_crop else camera_parameters["new_mtx"]
    distortion_coefficients = np.zeros((1, 5), dtype=np.float32)
    index = 'out' + str(camera_number)
    print(world_points[index].shape, image_points[index].shape, camera_matrix.shape, distortion_coefficients.shape)
    if with_crop:
        success, rotation_vector, translation_vector = cv.solvePnP(
            world_points[index], image_points[index], camera_matrix, distortion_coefficients
        )
    else:
        success, rotation_vector, translation_vector = cv.solvePnP(
            world_points[index], image_points[index], new_camera_matrix, distortion_coefficients
        )

    if not success:
        print("Failed to solve PnP", file=sys.stderr)
        sys.exit(1)

    # Convert the rotation vector to a rotation matrix using Rodrigues
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

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

    print(f"Camera {camera_number} extrinsic matrix:")

    pretty_print_matrix(extrinsic_matrix)

    size = args.size if args.size else 10
    plot_camera(extrinsic_matrix, all_camera_coordinates, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show the extrinsic matrix of a camera")

    parser.add_argument("camera_number", type=int, help="The camera number")

    undistortion_mode_group = parser.add_mutually_exclusive_group(required=True)
    undistortion_mode_group.add_argument(
        "-w",
        "--with-crop",
        action="store_true",
        help="Extract the extrinsic parameters from images undistorted with crop",
    )
    undistortion_mode_group.add_argument(
        "-wo",
        "--without-crop",
        action="store_true",
        help="Extract the extrinsic parameters from images undistorted without crop",
    )

    parser.add_argument("-s", "--size", type=int, help="The size of the plot")

    args = parser.parse_args()

    main(args)
