# coding=utf-8
from __future__ import print_function, division, absolute_import

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy.linalg as LA
import yaml
from scipy.interpolate import griddata
from projection_validation import *

import cv2
import numpy as np

SCALE_FACTOR = 1.5

def pc_to_img(pc, img, extrinsic_matrix, intrinsic_matrix, distortion):
    projection_points = undistort_projection(pc[:, :3], intrinsic_matrix, extrinsic_matrix)
    #print('pc3', projection_points.shape)
    # 裁切到图像平面
    projection_points = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])
    # NOTE: commented out below since camera fov larger than lidar
    # projection_points = projection_points[np.where(
    #     (projection_points[:, 0] > 0) &
    #     (projection_points[:, 0] < img.shape[1]) &
    #     (projection_points[:, 1] > 0) &
    #     (projection_points[:, 1] < img.shape[0])
    # )]
    #print('pc4', projection_points.shape)
    # scale
    img = cv2.resize(img, (int(img.shape[1] / SCALE_FACTOR) + 1, int(img.shape[0] / SCALE_FACTOR) + 1))
    projection_points[:, :2] /= SCALE_FACTOR

    board = np.zeros_like(img)

    # 提取边缘
    edge = np.uint8(np.absolute(cv2.Laplacian(
        cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F)))
    # board[..., 0] = board[..., 1] = edge ** 1.5
    board[...] = img

    # # colors = plt.get_cmap('gist_ncar_r')(projection_points[:, 3] / 255) * 2
    # colors = plt.get_cmap('gist_ncar_r')((30 - projection_points[:, 2]) / 30)
    # img_size = img.shape[:2][::-1]
    # grid_x, grid_y = np.mgrid[0:img_size[0]:1, 0:img_size[1]:1]
    # chs = [griddata(projection_points[:, 0:2], colors[:, 2 - idx], (grid_x, grid_y), method='linear').T for idx in range(3)]
    # board = np.stack(chs, axis=-1)

    # 反射率可视化
    # colors = plt.get_cmap('gist_ncar_r')(projection_points[:, 3] / 255) ** 2
    #colors = plt.get_cmap('gist_ncar_r')(projection_points[:, 3] / 255) ** 2
    colors = plt.get_cmap('magma_r')
    # colors = plt.get_cmap('gist_ncar_r')((30 - projection_points[:, 2]) / 30)
    # for idx in range(3):
    #     board[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), 2 - idx] = colors[:, idx] * 255
    norm_depth = projection_points[:, 2]/np.max(projection_points[:, 2])
    #print(projection_points.shape, 'final')
    board[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0])] =  (colors(norm_depth)[:,:3] * 255)[:,::-1]
    # board = board[120:, ...]

    # board = cv2.resize(board, dsize=(board.shape[1] // 2, board.shape[0] // 2))

    return board

if __name__=="__main__":
    
    data_fldr = os.listdir('/home/user/ACSC/calibration_data')
    assert len(data_fldr) == 1, "There should be exactly one folder of image/clouds from collect-calibrate, found {}".format(data_fldr)
    root = os.path.join('/home/user/ACSC/calibration_data', data_fldr[0])
    print('Root dir:', root)
    intrinsic_matrix = np.loadtxt(os.path.join(root, 'intrinsic'))
    distortion = np.loadtxt(os.path.join(root, 'distortion'))
    extrinsic_matrix = np.loadtxt(os.path.join(root, 'parameter/extrinsic'))
    
    for location in os.listdir('/data'):
        print('current location', location)
        base = os.path.join('/data', location)

        img_paths = sorted([f for f in os.listdir(base) if f.endswith('.png')])

        for img_path in img_paths:
            print('img_path', img_path)
            full_img_path = os.path.join(base, img_path)
            img = cv2.imread(full_img_path)
            pc = np.load(full_img_path.replace('png', 'npy'))
            #print('pc1', pc.shape)

            # 消除图像distortion
            img = cv2.undistort(img, intrinsic_matrix, distortion)

            # img = img[..., ::-1]
            # process pc
            # pc = pc[np.where(
            #     (pc[:, 0] > 0) &
            #     (abs(pc[:, 0]) < 100) &
            #     (pc[:, 2] > -3)
            # )]
            # print(((pc[:, 0] > 0) & (abs(pc[:, 0]) < 100) & (pc[:, 2] > -3)).sum(), pc.shape)
            # reprojection
            #print('pc2', pc.shape)
            board = pc_to_img(pc, img, extrinsic_matrix, intrinsic_matrix, distortion)

            # crop
            print(board.shape)
            board = board[50:350, 127:600+127]
            print(board.shape)
    
            cv2.imshow('Projection', board)
            cv2.waitKey(90)