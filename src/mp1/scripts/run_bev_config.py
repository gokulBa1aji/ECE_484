#!/usr/bin/env python3
import os
import json
import numpy as np


def main():
    # intrinsics (K) from /front_single_camera/camera_info
    K = np.array([
        [476.7030836014194, 0.0, 400.5],
        [0.0, 476.7030836014194, 300.5],
        [0.0, 0.0, 1.0]
    ])
    # extrinsics (R,t): base_footprint -> front_single_camera_link
    R = np.array([
        [0, -1, 0],
        [ 0, 0, 1],
        [ -1, 0, 0]
    ])
    t = np.array([0.160, -0.110, 1.546])

    # bird eye view image size (px); 
    # NOTE: DO NOT CHANGE THIS
    bev_img_height, bev_img_width = 600, 800

    # HYPERPARAMETERS: BEV rectangle configuration (m)
    # NOTE: Revert this to the original values for submission
    bev_height, bev_width, bev_margin= 10, 10, 3

    # px -> m conversion factor
    unit_conversion_factor = (bev_height/bev_img_height, bev_width/bev_img_width)
    bev_world_coords = np.float32([
        [bev_height, -bev_width/2, 0],
        [0, -bev_width/2, 0],
        [0, bev_width/2, 0],
        [bev_height, bev_width/2, 0],
    ])
    bev_world_coords[:, 0] += bev_margin

    # convert the bev_world_coords into pixel coordinates
    src = []
    for pt in bev_world_coords:
        ##### YOUR CODE STARTS HERE #####

        X_camera = R @ (pt.reshape(3,1) - t.reshape(3,1))
        x_img = K @ X_camera
        print("X_Image: ", x_img)
        u = x_img[0, 0] / x_img[2, 0]
        v = x_img[1, 0] / x_img[2, 0]
        src.append((u,v))


        # print("R1: ", R)
        # R_adjusted = np.block([
        #     [R, np.array([t]).T],
        #     [np.zeros((1, 3)), np.ones((1, 1))]
        # ])

        # pt_adjusted = np.vstack([pt.reshape(3, 1), [[1]]])

        # print("pt adjusted: ", pt_adjusted)

        # X_camera = R_adjusted @ pt_adjusted
        # print("X_camera: ", X_camera)

        # zeros_col = np.zeros((3, 1))

        # print("np.hstack: ", np.hstack((K, zeros_col)))

        # x_img = np.hstack((K, zeros_col)) @ X_camera

        # x_img = x_img / x_img[2, 0]

        # print("X_Image: ", x_img)
        # u = x_img[0] * unit_conversion_factor[1]
        # v = x_img[1] * unit_conversion_factor[0]

        # u = 476.7030836014194 * X_camera[0, 0] / X_camera[2, 0] + 400.5
        # v = 476.7030836014194 * X_camera[1, 0] / X_camera[2, 0] + 300.5

        # print("u, v:", u, v)

        # src.append((u,v))



        ##### YOUR CODE ENDS HERE #####
    src = np.float32(src)

    output = {
        "bev_world_dim": (bev_height, bev_width),
        "bev_from_base_link": bev_margin,
        "unit_conversion_factor": unit_conversion_factor,
        "src": src.tolist(),
    }
    # save config to json
    save_fn = 'data/bev_config.json'
    if not os.path.isdir('data/'):
        print(f"Data directory not found. Generating...")
        os.makedirs('data/', exist_ok=False)
    if os.path.isfile(save_fn):
        if input("File already exists. Overwrite? (y/n):").lower() != 'y':
            print("Exiting...")
            import sys
            sys.exit()
    with open(save_fn, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved BEV config to {save_fn}.")


if __name__ == "__main__":
    main()
