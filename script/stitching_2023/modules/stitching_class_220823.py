# ANTs
import fnmatch
from statistics import mean, median, variance, stdev
import os.path
import glob
import json
from scipy import ndimage as ndi
import subprocess as sp
import nibabel as nib
from scipy.ndimage import zoom
import copy
from multiprocessing import Pool
from PIL import Image
from scipy import ndimage
import datetime
import time
from joblib import Parallel, delayed
from MergeBrain_NeuN import WholeBrainCells
import ants
import tifffile
import SimpleITK as sitk
import os
import pandas as pd
import scipy.ndimage
import skimage.morphology
import sklearn.mixture
import cv2
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import sys
import os
from IPython.display import Image, display_png

import sys
sys.path.append('../')


plt.gray()


dt_local = np.dtype([
    ('local_x', 'f4'), ('local_y', 'f4'), ('local_z', 'f4'),
    ('structureness', 'f4'), ('blobness', 'f4'), ('intensity', 'f4'),
    ('size', 'u2'), ('padding', 'u2'), ('intensity2', 'f4'), ('intensity3', 'f4')
])


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


class stitching():
    def __init__(self, param, stitching_img_um):
        self.param = param

        self.param_dir = self.param["param_dir"]
        self.param_dir_nu = os.path.join(self.param_dir, "nu")
        self.param_nu_merge_path = os.path.join(
            self.param_dir, "param_merge.json")
        self.param_nu_clf = os.path.join(self.param_dir, "param_classify.json")

        # json読み込み(merge, FW, RV, clf)
        with open(self.param_nu_merge_path) as f:
            self.param_nu_merge = json.load(f)
        with open(self.param_nu_merge['HDoG_paramfile']["FW"]) as f:
            self.param_nu_FW = json.load(f)
        with open(self.param_nu_merge['HDoG_paramfile']["RV"]) as f:
            self.param_nu_RV = json.load(f)
        with open(self.param_nu_clf) as f:
            self.param_clf = json.load(f)

        self.src_base_dir_FW = self.param['src_basedir']["FW"]
        self.src_base_dir_RV = self.param['src_basedir']["RV"]

        self.x_pixel = self.param['input_image_info']['width']
        self.y_pixel = self.param['input_image_info']['height']
        self.x_overlap = self.param['input_image_info']['left_margin']
        self.y_overlap = self.param['input_image_info']['top_margin']
        self.FW_boundary = self.param['merge_info']["boundary_fname"]["FW_boundary"]
        self.RV_boundary = self.param['merge_info']["boundary_fname"]["RV_boundary"]
        self.FW_boundary_name = self.param['merge_info']["boundary_fname"]["FW_name"]
        self.RV_boundary_name = self.param['merge_info']["boundary_fname"]["RV_name"]

        self.scale_x = self.param["coordinate_info"]["scale_x"]
        self.scale_y = self.param["coordinate_info"]["scale_y"]
        self.scale_z = self.param["coordinate_info"]["scale_z"]

        # FW_img_rotation
        # stackの中での左右flip
        self.FW_stack_flip_x = self.param["merge_info"]['flip_rot']["FW"]["stack_flipX"]
        self.FW_stack_flip_y = self.param["merge_info"]['flip_rot']["FW"]["stack_flipY"]
        # 画像全体のstackの順番
        self.FW_merge_flip_x = self.param["merge_info"]['flip_rot']["FW"]["merge_flipX"]
        self.FW_merge_flip_y = self.param["merge_info"]['flip_rot']["FW"]["merge_flipY"]
        # 画像右回転or左回転
        self.rotCW_FW = self.param["merge_info"]['flip_rot']["FW"]["rotCW"]
        self.rotCCW_FW = self.param["merge_info"]['flip_rot']["FW"]["rotCCW"]

        # RV_img_rotation
        # stackの中での左右flip
        self.RV_stack_flip_x = self.param["merge_info"]['flip_rot']["RV"]["stack_flipX"]
        self.RV_stack_flip_y = self.param["merge_info"]['flip_rot']["RV"]["stack_flipY"]
        # 画像全体のstackの順番
        # self.RV_merge_flip_x != self.FW_merge_flip_x
        self.RV_merge_flip_x = self.param["merge_info"]['flip_rot']["RV"]["merge_flipX"]
        # self.RV_merge_flip_y != self.FW_merge_flip_y
        self.RV_merge_flip_y = self.param["merge_info"]['flip_rot']["RV"]["merge_flipY"]
        # 画像右回転or左回転
        self.rotCW_RV = self.param["merge_info"]['flip_rot']["RV"]["rotCW"]
        self.rotCCW_RV = self.param["merge_info"]['flip_rot']["RV"]["rotCCW"]

        # 画像のないところのintensity やや高めに！！
        self.bcg_intensity = self.param["bcg_intensity"]

        # 書き出しdirectoryの作成
        self.stitching_dst_path = self.param['dst_basedir']
        make_dir(self.stitching_dst_path)
        self.stitching_check_path = os.path.join(
            self.stitching_dst_path, "check")
        make_dir(self.stitching_check_path)

        # raw dataのpath
        self.src_base_dir_FW = self.param['src_basedir']["FW"]
        self.src_base_dir_RV = self.param['src_basedir']["RV"]

        # x,yのstackの数
        self.y_list_num = len(os.listdir(self.src_base_dir_FW))
        self.x_list_num = len(os.listdir(os.path.join(
            self.src_base_dir_FW, os.listdir(self.src_base_dir_FW)[0])))

        # 画像全体でのstackの順番
        if self.FW_merge_flip_x == 0:
            self.list_x_FW = list(range(self.x_list_num))
        if self.FW_merge_flip_x == 1:
            self.list_x_FW = list(reversed(range(self.x_list_num)))
        if self.FW_merge_flip_y == 0:
            self.list_y_FW = list(range(self.y_list_num))
        if self.FW_merge_flip_y == 1:
            self.list_y_FW = list(reversed(range(self.y_list_num)))
        if self.RV_merge_flip_x == 0:
            self.list_x_RV = list(range(self.x_list_num))
        if self.RV_merge_flip_x == 1:
            self.list_x_RV = list(reversed(range(self.x_list_num)))
        if self.RV_merge_flip_y == 0:
            self.list_y_RV = list(range(self.y_list_num))
        if self.RV_merge_flip_y == 1:
            self.list_y_RV = list(reversed(range(self.y_list_num)))

        # core_short
        self.core_short = self.param["xy_stitching"]["core_short"]
        self.core_short_half = int(self.core_short/2)

        # core_long
        self.core_long_edge = self.param["xy_stitching"]["core_long_edge"]
        self.right_core_long = self.y_pixel-self.core_long_edge*2
        self.down_core_long = self.x_pixel-self.core_long_edge*2
        self.right_core_long_half = int(self.right_core_long/2)
        self.down_core_long_half = int(self.down_core_long/2)

        # target_short
        self.target_short_half = self.core_short_half+75
        assert self.target_short_half <= self.x_overlap, "self.target_short_half > self.x_overlap"
        assert self.target_short_half <= self.y_overlap, "self.target_short_half > self.y_overlap"

        # target_long
        self.left_target_long_half = self.right_core_long_half+75
        self.up_target_long_half = self.down_core_long_half+75

        # FW_RV_stitching
        self.FW_RV_xy_scan_range = 100  # xy_scan_range(100*2+1pixel scanする)
        self.FW_RV_z_scan_range = 100  # z_scan_range(100pixel scanする)
        self.FW_RV_stitching_scale = 5  # must be int*zstep
        self.z_step = int(self.FW_RV_stitching_scale/self.scale_z)
        self.FW_ants_maisuu = 5
        self.RV_ants_maisuu = 151
        self.large_transform_path = self.stitching_dst_path+"/large_transform.mat"
        self.RV_maisuu_path = self.stitching_dst_path+"/RV_maisuu.txt"
        self.RV_large_moved_center_path = self.stitching_dst_path + \
            "/check/RV_large_moved_center.tif"

        make_dir(os.path.join(self.stitching_dst_path, "FW/left/"))
        make_dir(os.path.join(self.stitching_dst_path, "FW/up/"))
        make_dir(os.path.join(self.stitching_dst_path, "RV/left/"))
        make_dir(os.path.join(self.stitching_dst_path, "RV/up/"))
        make_dir(os.path.join(self.stitching_dst_path, "FW_RV/"))
        make_dir(os.path.join(self.stitching_check_path, "RV_merge_boundary"))
        make_dir(os.path.join(self.stitching_check_path, "FW_target"))
        make_dir(os.path.join(self.stitching_check_path, "RV_moving"))
        make_dir(os.path.join(self.stitching_check_path, "RV_moved"))
        make_dir(os.path.join(self.stitching_check_path, "FW_RVresult"))
        make_dir(os.path.join(self.stitching_check_path, "FW_RVtarget"))

        # make scalemerage image
        # 1um,5umスケールでのstackの一辺のpixel
        self.scalemerge_um = stitching_img_um
        self.stack_1um_x = int(self.x_pixel*self.scale_x)
        self.stack_1um_y = int(self.y_pixel*self.scale_y)
        self.x_overlap_1um = self.x_overlap*self.scale_x
        self.y_overlap_1um = self.y_overlap*self.scale_y
        self.stack_5um_x = int(self.x_pixel*self.scale_x/5)
        self.stack_5um_y = int(self.y_pixel*self.scale_y/5)
        self.stack_scalemerge_um_x = int(
            self.x_pixel*self.scale_x/self.scalemerge_um)
        self.stack_scalemerge_um_y = int(
            self.y_pixel*self.scale_y/self.scalemerge_um)

        self.stitched_images_path = self.stitching_dst_path + \
            "/stitched_images_{}um_scale/".format(self.scalemerge_um)
        make_dir(self.stitched_images_path)
        self.saving_path_before_rotation = self.stitching_dst_path + \
            "/RV_merged_before_rotation_{}um_scale/".format(self.scalemerge_um)
        make_dir(self.saving_path_before_rotation)
        self.saving_path_after_rotation = self.stitching_dst_path + \
            "/RV_merged_after_rotation_{}um_scale/".format(self.scalemerge_um)
        make_dir(self.saving_path_after_rotation)

        # point local to global
        self.global_FW_dst = os.path.join(self.stitching_dst_path, "global/FW")
        make_dir(self.global_FW_dst)
        self.global_RV_dst = os.path.join(self.stitching_dst_path, "global/RV")
        make_dir(self.global_RV_dst)
        self.density_img_path = os.path.join(
            self.stitching_dst_path, "{}um_scale_density_img".format(stitching_img_um))
        make_dir(self.density_img_path)

        # rasen_xy
        rasen_x = []
        for i in range(10):
            rasen_x = rasen_x + list(range(-1*i, i+1))
            rasen_x = rasen_x + [i+1]*(i+1)*2
            rasen_x = rasen_x + list(range(i, -1*(i+1), -1))
            rasen_x = rasen_x + [-1*(i+1)]*(i+1)*2
        rasen_y = [0]
        for i in range(10):
            rasen_y = rasen_y + list(range(-1*i, i+1))
            rasen_y = rasen_y + [i+1]*((i+1)*2+1)
            rasen_y = rasen_y + list(range(i, -1*(i+1), -1))
            rasen_y = rasen_y + [-1*(i+1)]*((i+1)*2+1)
        rasen_xy = []
        for i in range(300):
            rasen_xy.append([rasen_y[i], rasen_x[i]])
        self.rasen_xy = np.array(
            rasen_xy)+np.array([int(self.x_list_num/2), int(self.y_list_num/2)])

    ###################### 基本的動作###############################
    def import_bin(self, src_path, FWorRV, _dtype=np.uint16):
        if os.path.exists(src_path):
            # print("src_path is {}".format(src_path))
            img = np.fromfile(src_path, dtype=_dtype)
            img = img.reshape(2048, 2060)
            img = img[:2048, :2048]
        if not os.path.exists(src_path):
            # print("no data {}".format(src_path))
            img = np.zeros((2048, 2048))
        return img

    def import_img_xyz(self, x_num, y_num, z_num, FWorRV):
        if FWorRV == "FW":
            xx, yy = self.list_x_FW[x_num], self.list_y_FW[y_num]
            z_name = str(z_num).zfill(8)+".bin"
            y_path_list = sorted(os.listdir(self.src_base_dir_FW))
            x_path_list = sorted(os.listdir(os.path.join(
                self.src_base_dir_FW, y_path_list[yy])))
            src_path = self.src_base_dir_FW + "/" + \
                y_path_list[yy] + "/" + x_path_list[xx] + "/" + z_name
        if FWorRV == "RV":
            xx, yy = self.list_x_RV[x_num], self.list_y_RV[y_num]
            z_name = str(z_num).zfill(8)+".bin"
            y_path_list = sorted(os.listdir(self.src_base_dir_RV))
            x_path_list = sorted(os.listdir(os.path.join(
                self.src_base_dir_RV, y_path_list[yy])))
            src_path = self.src_base_dir_RV + "/" + \
                y_path_list[yy] + "/" + x_path_list[xx] + "/" + z_name
        img = self.import_bin(src_path, FWorRV)
        if FWorRV == "FW":
            if self.rotCW_FW == 1:
                img = np.rot90(img, k=1)
            if self.rotCCW_FW == 1:
                img = np.rot90(img, k=-1)
            if self.FW_stack_flip_x == 1:
                img = np.fliplr(img)
            if self.FW_stack_flip_y == 1:
                img = np.flipud(img)
        if FWorRV == "RV":
            if self.rotCW_RV == 1:
                img = np.rot90(img, k=1)
            if self.rotCCW_RV == 1:
                img = np.rot90(img, k=-1)
            if self.RV_stack_flip_x == 1:
                img = np.fliplr(img)
            if self.RV_stack_flip_y == 1:
                img = np.flipud(img)
        return img

    def norm(self, x_num, y_num, z_num, FWorRV):
        src_img = self.import_img_xyz(x_num, y_num, z_num, FWorRV)
        dilation_l_img = scipy.ndimage.filters.uniform_filter(
            scipy.ndimage.morphology.grey_dilation(
                src_img, size=50, mode="reflect").astype(np.float32),
            size=100, mode="reflect", cval=0)
        erosion_l_img = scipy.ndimage.filters.uniform_filter(
            scipy.ndimage.morphology.grey_erosion(
                src_img, size=50, mode="reflect").astype(np.float32),
            size=100, mode="reflect", cval=0)

        intensity = src_img.astype(np.float32)
        norm_img = (intensity >= self.bcg_intensity) * (intensity -
                                                        erosion_l_img) / (dilation_l_img - erosion_l_img)
        return norm_img

    def normalize_img(self, img, Min, Max):
        img[img < Min] = Min
        img[img > Max] = Max
        img = img-Min
        img = img*(1000/Max)
        return img

    ##################### rotationの確認##############################
    def check_img(self, FWorRV):
        if FWorRV == "FW":
            ZZ = self.FW_boundary
        if FWorRV == "RV":
            ZZ = self.RV_boundary
        micro = 5
        downscale_ratio = micro/self.scale_x
        down_y = int(self.y_pixel*self.y_list_num/downscale_ratio)
        print(down_y)
        down_x = int(self.x_pixel*self.x_list_num/downscale_ratio)
        img = np.zeros((down_y, down_x))
        y_step = int(self.y_pixel/downscale_ratio)
        print(y_step)
        x_step = int(self.x_pixel/downscale_ratio)

        for YY in range(self.y_list_num):
            for XX in range(self.x_list_num):
                a = self.norm(XX, YY, ZZ, FWorRV)
                b = cv2.resize(a, dsize=None, fx=self.scale_x /
                               micro, fy=self.scale_y/micro)
                down_y_stack, down_x_stack = np.shape(b)
                # bb =np.split(np.array(np.split(a,int(self.x_pixel/downscale_ratio),1)),int(self.y_pixel/downscale_ratio),1)
                img[y_step*YY:y_step*YY+down_y_stack, x_step *
                    XX:x_step*XX+down_x_stack] = b*(b > 0)
        save_img_path = os.path.join(
            self.stitching_check_path, "{}_check_before.tif".format(FWorRV))
        tifffile.imsave(save_img_path, (img*100).astype(np.uint16))
        return

    ##################### x-y-stitching##############################
    def compare_to_left(self, XX, YY, ZZ, FWorRV):
        if FWorRV == "FW":
            list_x = self.list_x_FW
            list_y = self.list_y_FW
        if FWorRV == "RV":
            list_x = self.list_x_RV
            list_y = self.list_y_RV
        scan_step = 3
        Left_img = self.norm(list_x[XX-1], list_y[YY], ZZ, FWorRV)
        Left_target_ymin = int(self.y_pixel/2)-self.left_target_long_half
        Left_target_ymax = int(self.y_pixel/2)+self.left_target_long_half
        Left_target_xmin = self.x_pixel-self.x_overlap-self.target_short_half
        Left_target_xmax = self.x_pixel-self.x_overlap+self.target_short_half
        Left_target = Left_img[Left_target_ymin:Left_target_ymax,
                               Left_target_xmin: Left_target_xmax]
        # print(Left_target_ymin,Left_target_ymax, Left_target_xmin, Left_target_xmax)

        Right_core_ymin = int(self.y_pixel/2)-self.right_core_long_half
        Right_core_ymax = int(self.y_pixel/2)+self.right_core_long_half
        Right_core_xmin = self.x_overlap-self.core_short_half
        Right_core_xmax = self.x_overlap+self.core_short_half
        # print(Right_core_ymin,Right_core_ymax ,Right_core_xmin,Right_core_xmax)
        # TODO他のsearch範囲にも対応できるようにする！
        Right_core_images = [self.norm(list_x[XX], list_y[YY], ZZ+z_search-25, FWorRV)[
            Right_core_ymin:Right_core_ymax, Right_core_xmin:Right_core_xmax] for z_search in range(51)]
        R = []
        for i, template_img in enumerate(Right_core_images):
            # 画像の検索（Template Matching）
            # print(i)
            result = cv2.matchTemplate(
                Left_target, template_img, cv2.TM_CCORR_NORMED)
            # 検索結果の信頼度と位置座標の取得
            R.append(cv2.minMaxLoc(result))
        Sz = (np.argmax(np.array(R).T[1]))
        Sx, Sy = R[Sz][3]
        z_diff, y_diff, x_diff = Sz-25, Sy-75, Sx-75
        return np.array([x_diff, y_diff, z_diff, np.max(np.array(R).T[1])])

    def compare_to_up(self, XX, YY, ZZ, FWorRV):
        if FWorRV == "FW":
            list_x = self.list_x_FW
            list_y = self.list_y_FW
        if FWorRV == "RV":
            list_x = self.list_x_RV
            list_y = self.list_y_RV

        scan_step = 3

        Up_img = self.norm(list_x[XX], list_y[YY-1], ZZ, FWorRV)
        Up_core_ymin = self.y_pixel-self.y_overlap-self.target_short_half
        Up_core_ymax = self.y_pixel-self.y_overlap+self.target_short_half
        Up_core_xmin = int(self.x_pixel/2)-self.up_target_long_half
        Up_core_xmax = int(self.x_pixel/2)+self.up_target_long_half
        Up_target = Up_img[Up_core_ymin:Up_core_ymax,
                           Up_core_xmin:Up_core_xmax]
        # print(Up_core_ymin,Up_core_ymax,Up_core_xmin,Up_core_xmax)

        Down_core_ymin = self.y_overlap-self.core_short_half
        Down_core_ymax = self.y_overlap+self.core_short_half
        Down_core_xmin = int(self.x_pixel/2)-self.down_core_long_half
        Down_core_xmax = int(self.x_pixel/2)+self.down_core_long_half
        Down_core_images = [self.norm(list_x[XX], list_y[YY], ZZ+z_search-25, FWorRV)[
            Down_core_ymin:Down_core_ymax, Down_core_xmin:Down_core_xmax] for z_search in range(51)]

        R = []
        for i, template_img in enumerate(Down_core_images):
            # 画像の検索（Template Matching）
            # print(i)
            result = cv2.matchTemplate(
                Up_target, template_img, cv2.TM_CCORR_NORMED)
            # 検索結果の信頼度と位置座標の取得
            R.append(cv2.minMaxLoc(result))
        Sz = (np.argmax(np.array(R).T[1]))
        Sx, Sy = R[Sz][3]
        z_diff, y_diff, x_diff = Sz-25, Sy-75, Sx-75
        return np.array([x_diff, y_diff, z_diff, np.max(np.array(R).T[1])])
