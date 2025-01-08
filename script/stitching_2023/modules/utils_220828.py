# ANTs
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
import joblib, json, sys, os
from IPython.display import Image, display_png

import sys
sys.path.append('../')

import json, glob, os.path


from joblib import Parallel, delayed
import numpy as np
import tifffile
import os,os.path,fnmatch
import time
import datetime
from scipy import ndimage
plt.gray()
from PIL import Image
from multiprocessing import Pool
from statistics import mean, median,variance,stdev
import copy
from scipy.ndimage import zoom

import nibabel as nib
import subprocess as sp
from scipy import ndimage as ndi
import shutil
import re
import gc

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def tif2nii(np_img,nii_path,nii_voxel_unit_xy,nii_voxel_unit_z):
    nii_img = nib.Nifti1Image(np.swapaxes(np_img,0,2), affine=None)
    aff = np.diag([nii_voxel_unit_xy,nii_voxel_unit_xy,nii_voxel_unit_z,1])
    nii_img.header.set_qform(aff, code=2)
    nii_img.to_filename(nii_path)
    return

def tif2nii_no_save(np_img,nii_voxel_unit_x,nii_voxel_unit_y,nii_voxel_unit_z):
    nii_img = nib.Nifti1Image(np.swapaxes(np_img,0,2), affine=None)
    aff = np.diag([nii_voxel_unit_x,nii_voxel_unit_y,nii_voxel_unit_z,1])
    nii_img.header.set_qform(aff, code=2)
    del np_img, aff
    gc.collect()
    return nii_img

def make_density_image(list_x,list_y,list_z,size_x,size_y,size_z):
    depth,height,width = size_z,size_y,size_x
    density_img,_ = np.histogramdd(
        np.vstack([list_z,list_y,list_x]).T,
        bins=(depth, height, width),
        range=[(0,depth-1),(0,height-1),(0,width-1)]
    )
    return density_img

def normalize_image(img,Min,Max):
    img[img<Min]=Min
    img[img>Max]=Max
    img = img-Min
    img = img*(1000/Max)
    return img


def line_formula_x(point1,point2):
    x1,y1 = point1
    x2,y2 = point2
    if y2 != y1:
        a = (x2-x1)/(y2-y1)
        b = (-y1*x2 + y2*x1)/(y2-y1)
        return a,b
    if y1 == y2:
        #print("error!!_line_formula_x y1=y2")
        return 0


def line_formula_y(point1,point2):
    x1,y1 = point1
    x2,y2 = point2
    if x2 != x1:
        a = (y2-y1)/(x2-x1)
        b = (-x1*y2 + x2*y1)/(x2-x1)
        return a,b
    if x1 == x2:
        #print("error!!_line_formula_y x1=x2")
        return 0
    
dt_local = np.dtype([
    ('local_x', 'f4'), ('local_y', 'f4'), ('local_z', 'f4'),
    ('structureness', 'f4'), ('blobness', 'f4'),('intensity', 'f4'),
    ('size', 'u2'),('padding', 'u2'),('intensity2', 'f4'),('intensity3', 'f4')
])