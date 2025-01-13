#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""Overview:
  Multi-channel verification by intensity ratio.

Usage:
  MultiChannelVerification-rank.py PARAM_FILE [-p NUM_CPUS]

Options:
  -h --help      Show this screen.
  --version      Show version.
  -p NUM_CPUS    Number of CPUs to be used. [default: 10]
"""
import gc
import numpy as np
import pandas as pd
import joblib
import os.path
from docopt import docopt
from MergeBrain_NeuN import WholeBrainCells
from MergeBrain_NeuN import WholeBrainImages
from evaluation import *
from HDoG_classifier import *
from HalfBrainCells_NeuN import *
from scipy.stats import rankdata
from scipy.stats import ranksums
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
# from skimage import exposure
# np.seterr(divide='ignore', invalid='ignore')

# dt_classified = np.dtype([
#    ('intensity2', 'f4'),('intensity3', 'f4')
# ])

dt_local = np.dtype([
    ('local_x', 'f4'), ('local_y', 'f4'), ('local_z', 'f4'),
    ('structureness', 'f4'), ('blobness', 'f4'), ('intensity', 'f4'),
    ('size', 'u2'), ('padding', 'u2'), ('intensity2', 'f4'), ('intensity3', 'f4')
])

dt_annotated_classified = np.dtype([
    ('local_x', 'f4'), ('local_y', 'f4'), ('local_z', 'f4'),
    ('stitched_x', 'f4'), ('stitched_y', 'f4'), ('stitched_z', 'f4'),
    ('mapped_x', 'f4'), ('mapped_y', 'f4'), ('mapped_z', 'f4'),
    ('atlas_id', 'u2'), ('is_positive','f4'),('threshold','f4'),('composite_value','f4') # ここのis_positiveはinitial classifyによるものにする
])

dt_annotated_classified_intensity = np.dtype([
    ('local_x', 'f4'), ('local_y', 'f4'), ('local_z', 'f4'),
    ('stitched_x', 'f4'), ('stitched_y', 'f4'), ('stitched_z', 'f4'),
    ('mapped_x', 'f4'), ('mapped_y', 'f4'), ('mapped_z', 'f4'),
    ('atlas_id', 'u2'), ('is_positive','f4'), ('intensity','f4'),('structureness','f4'),
    ('intensity2','f4'),('intensity3','f4'), # ここのis_positiveはinitial classifyによるものにする
    ('norm_intensity_2','f4'), ('norm_intensity_3','f4')
])

dt_annotated_init = np.dtype([
    ('local_x', 'f4'), ('local_y', 'f4'), ('local_z', 'f4'),
    ('stitched_x', 'f4'), ('stitched_y', 'f4'), ('stitched_z', 'f4'),
    ('mapped_x', 'f4'), ('mapped_y', 'f4'), ('mapped_z', 'f4'),
    ('atlas_id', 'u2'), ('is_positive','f4'),('intensity','f4'),('structureness','f4'), # ここのis_positiveはinitial classifyによるもの
    ('intensity_2','f4'), ('intensity_3','f4')
])

half_ROI_size_xy = 20 #40 #30 #20 # 12 #20  # 12 #25
half_ROI_size_z = int(round(half_ROI_size_xy * (0.65/2.5))) #2 #5 #int(round(half_ROI_size_xy * (0.65/2.5)))  # 3 pix


def save_intensities(
        xyname,
        resultfile_nucl,
        imagestack_nucl,
        clf, dst_basedir,
        left_stack_xmin, offset_right_NeuN, offset_right_Iba1, offset_left_NeuN, offset_left_Iba1, FWRV):

    #data_local_nucl = np.fromfile(resultfile_nucl, dtype=dt_local)
    if os.path.exists(resultfile_nucl.replace("candidate_nu_", "nu_R_/result")):
        #data_local_nucl = joblib.load(resultfile_nucl.replace("candidate_nu_", "nu_R_/result_nuc_classified").replace(".bin", ".pkl"))
        #data_local_nucl = np.array(data_local_nucl, dtype=dt_annotated_classified)
        #data_local_nucl = np.fromfile(resultfile_nucl.replace("candidate_nu_", "nu_R_/result_nuc_classified"), dtype = dt_annotated_classified)
        data_local_nucl = np.fromfile(resultfile_nucl.replace("candidate_nu_", "nu_R_/result"), dtype = dt_annotated_init)
    else:
        print("Stitched File not found:" + str(resultfile_nucl.replace("candidate_nu_", "nu_R_/result")))
        return

    if 1==0:
        data_intensity =  np.zeros((data_local_nucl.shape[0],), dtype=dt_annotated_classified_intensity) 
        data_intensity["local_x"] = data_local_nucl["local_x"]
        data_intensity["local_y"] = data_local_nucl["local_y"]
        data_intensity["local_z"] = data_local_nucl["local_z"]

        data_intensity["stitched_x"] = data_local_nucl["stitched_x"]
        data_intensity["stitched_y"] = data_local_nucl["stitched_y"]
        data_intensity["stitched_z"] = data_local_nucl["stitched_z"]

        data_intensity["mapped_x"] = data_local_nucl["mapped_x"]
        data_intensity["mapped_y"] = data_local_nucl["mapped_y"]
        data_intensity["mapped_z"] = data_local_nucl["mapped_z"]

        data_intensity["atlas_id"] = data_local_nucl["atlas_id"]
        data_intensity["is_positive"] = data_local_nucl["is_positive"]
        data_intensity["intensity2"] = 0.0
        data_intensity["intensity3"] = 0.0

    if 1==1:
        # data_local_nucl = data_local_nucl[data_local_nucl["atlas_id"] > 0.0] # annotationが変わると変化するので却下。
        data_local_nucl = data_local_nucl[(data_local_nucl["stitched_x"] != 0.0) & (data_local_nucl["stitched_y"] != 0.0) & (data_local_nucl["is_positive"] > 0.0)]

        data_intensity =  np.zeros((data_local_nucl.shape[0],), dtype=dt_annotated_classified_intensity) 
        
        data_intensity["local_x"] = data_local_nucl["local_x"]
        data_intensity["local_y"] = data_local_nucl["local_y"]
        data_intensity["local_z"] = data_local_nucl["local_z"]

        data_intensity["stitched_x"] = data_local_nucl["stitched_x"]
        data_intensity["stitched_y"] = data_local_nucl["stitched_y"]
        data_intensity["stitched_z"] = data_local_nucl["stitched_z"]

        data_intensity["mapped_x"] = data_local_nucl["mapped_x"]
        data_intensity["mapped_y"] = data_local_nucl["mapped_y"]
        data_intensity["mapped_z"] = data_local_nucl["mapped_z"]

        data_intensity["atlas_id"] = data_local_nucl["atlas_id"]
        data_intensity["is_positive"] = data_local_nucl["is_positive"]
        data_intensity["intensity"] = data_local_nucl["intensity"]
        data_intensity["structureness"] = data_local_nucl["structureness"]

        data_intensity["intensity2"] = 0.0
        data_intensity["intensity3"] = 0.0
        data_intensity["norm_intensity_2"] = data_local_nucl["intensity_2"]
        data_intensity["norm_intensity_3"] = data_local_nucl["intensity_3"]

    del data_local_nucl
    gc.collect()

    if data_intensity.shape[0] == 0:
        return

    print(FWRV + "_xyname: " + xyname[1] + "_" + xyname[0])
    # print("xyname[0]: " + xyname[0]) # xyname[0]は、xnameのことであった。

    offset_NeuN = offset_right_NeuN
    offset_Iba1 = offset_right_Iba1

    if FWRV == "FW":

        if int(xyname[0]) >= left_stack_xmin:
            # left
            # print("Left stack")
            offset_NeuN = offset_left_NeuN
            offset_Iba1 = offset_left_Iba1
        else:
            offset_NeuN = offset_right_NeuN
            offset_Iba1 = offset_right_Iba1
            # print("Right stack")
    if FWRV == "RV":
        if int(xyname[0]) < left_stack_xmin:
            # left
            # print("Left stack")
            offset_NeuN = offset_left_NeuN
            offset_Iba1 = offset_left_Iba1
        else:
            offset_NeuN = offset_right_NeuN
            offset_Iba1 = offset_right_Iba1

    coordinate = np.stack([data_intensity["local_z"],
                          data_intensity["local_y"], data_intensity["local_x"]], axis=1)
    coordinate = np.round(coordinate).astype(int)
    # print(coordinate.T[0]) [  9.   8.   8. ... 692. 691. 692.]

    #_X = get_X_3d(data_local_nucl) #_Xはいらない。もうclassify済みなので。
    #pred1 = clf.predict(_X)

    print("Nuclear+ num " + str(len(coordinate)))
    #print("Nuclear+     " + str((np.sum(pred1)*100)/len(data_local_nucl)) + " %")



    # np.zeros_like(data_local_nucl["intensity2"])
    #data_local_nucl["intensity2"] = np.zeros(
    #    len(data_local_nucl["intensity2"]), dtype=np.float32)
    #data_local_nucl["intensity3"] = np.zeros(
    #    len(data_local_nucl["intensity3"]), dtype=np.float32)

    #classify_posi = ~np.all(coordinate == [0, 0, 0], axis=1)*pred1  # classify

    per_voxel =  1 / ((2*half_ROI_size_xy + 1)*(2*half_ROI_size_xy + 1)*(2*half_ROI_size_z + 1))

    # CCLの幅に合わせて、classify点を参照する。
    # CCLの幅を無視
    band = 76 #136
    z_overlap = 8

    src_img = np.zeros((band, 2048, 2048), dtype=np.uint16)
    src_img_NeuN = np.zeros((band, 2048, 2048), dtype=np.uint16)
    src_img_Iba1 = np.zeros((band, 2048, 2048), dtype=np.uint16)
    #src_img_CCL = np.zeros((band, 2048, 2048), dtype=np.uint16)

    z0 = int(imagestack_nucl.list_zs[0])
    for z1 in imagestack_nucl.list_zs[::(band - 2 * z_overlap)]:# 0, 32, 64, ... if band = 32
        print("z = " + str(z1-z0))
        src_img_NeuN = np.zeros((band, 2048, 2048), dtype=np.uint16)
        src_img_Iba1 = np.zeros((band, 2048, 2048), dtype=np.uint16)

        # classifyでTrueのとき
        indices = (coordinate.T[0] >= z1-z0 + z_overlap) * (coordinate.T[0] < z1-z0 +  band - z_overlap)
        indices_num = np.array(np.where(
            (coordinate.T[0] >= z1-z0 + z_overlap) * (coordinate.T[0] < z1-z0 +  band - z_overlap))[0])
        print("Cell num " + str(np.sum(indices)))
        if np.sum(indices) > 0:
            for z, imgfile in enumerate(imagestack_nucl.list_imagefiles_no_dummy[z1-z0:z1-z0+band]):
                # z範囲外のときを排除
                if z1-z0+z < len(imagestack_nucl.list_zs):
                    src_img_NeuN[z, :, :] = imgfile.load_image_NeuN(
                        offset=offset_NeuN[2])
                    src_img_Iba1[z, :, :] = imgfile.load_image_Iba1(
                        offset=offset_Iba1[2])

            temp2 = coordinate[indices].copy()

            # classifyにより、核でないやつは、0,0,0になるらしい。
            for i2, coor in enumerate(temp2):
                temp_indice = indices_num[i2]
                temp_value = rank_average_simple(
                    coor, 2, src_img_NeuN, src_img_Iba1, z1-z0, offset_NeuN, offset_Iba1)
                data_intensity["intensity2"][temp_indice] = (1 - temp_value[0]*per_voxel).astype(
                    np.float32)
                data_intensity["intensity3"][temp_indice] = (1 - temp_value[1]*per_voxel).astype(
                    np.float32)

                #data_local_nucl["intensity2"][temp_indice] = temp_value[0].astype(
                #    np.float32)
                #data_local_nucl["intensity3"][temp_indice] = temp_value[1].astype(
                #    np.float32)

            hy = np.sum(data_intensity["intensity2"][indices] > 0.66) # > 0.7
            hd = len(temp2)
            ht = hy/hd
            print("NeuN+ " + str(ht*100) + " %")
            hy = np.sum(data_intensity["intensity3"][indices] > 0.88) # > 0.95
            hd = len(temp2)
            ht = hy/hd
            print("Iba1+ " + str(ht*100) + " %")

    data_intensity_array = np.array(data_intensity, dtype=dt_annotated_classified_intensity)
    print(data_intensity_array)

    #joblib.dump(data_intensity_array, os.path.join(
    #    dst_basedir, "{}_{}.bin".format(xyname[1], xyname[0])))
    data_intensity_array.tofile(os.path.join(dst_basedir, "{}_{}.bin".format(xyname[1], xyname[0])))
    #print( os.path.join(dst_basedir, "{}_{}.bin".format(xyname[1], xyname[0])))

    if 1 == 0:
        tifffile.imsave(
            "/export3/Imaging/Axial/Neurorology/test3/test/nuc.tif",
            src_img[15]
        )
        tifffile.imsave(
            "/export3/Imaging/Axial/Neurorology/test3/test/NeuN.tif",
            src_img_NeuN[15]
        )
        tifffile.imsave(
            "/export3/Imaging/Axial/Neurorology/test3/test/Iba1.tif",
            src_img_Iba1[15]
        )
        tifffile.imsave(
            "/export3/Imaging/Axial/Neurorology/test3/test/CCL.tif",
            src_img_CCL
        )
    gc.collect()

# Gaussian sigma values
sigma_xy = 6#2
sigma_z = sigma_xy / (2.5 / 0.65)

def rank_average_simple (coordi, radius, img16_NeuN3, img16_Iba13, z_posi_offset, offset_NeuN, offset_Iba1):  # 体積
    rank_NeuN_average = (2*half_ROI_size_xy + 1)*(2*half_ROI_size_xy + 1)*(2*half_ROI_size_z + 1)
    rank_Iba1_average = (2*half_ROI_size_xy + 1)*(2*half_ROI_size_xy + 1)*(2*half_ROI_size_z + 1)

    coordi[0] = coordi[0] - z_posi_offset  # 補正

    xmin = int(coordi[2]-half_ROI_size_xy)
    xmax = int(coordi[2]+half_ROI_size_xy)
    ymin = int(coordi[1]-half_ROI_size_xy)
    ymax = int(coordi[1]+half_ROI_size_xy)
    zmin = int(coordi[0]-half_ROI_size_z)
    zmax = int(coordi[0]+half_ROI_size_z)

    xmin_s = half_ROI_size_xy - radius 
    xmax_s = half_ROI_size_xy + 1 + radius
    ymin_s = half_ROI_size_xy - radius
    ymax_s = half_ROI_size_xy + 1 + radius
    zmin_s = half_ROI_size_z - 1 # とりあえず、zはradius = 1にしておく
    zmax_s = half_ROI_size_z + 1 + 1

    # Extract subset of images only once
    subset_NeuN3 = img16_NeuN3[zmin:zmax, (ymin + offset_NeuN[1]):(ymax + offset_NeuN[1]), (xmin + offset_NeuN[0]):(xmax+ offset_NeuN[0])]
    subset_Iba13 = img16_Iba13[zmin:zmax, (ymin + offset_Iba1[1]):(ymax + offset_Iba1[1]), (xmin + offset_Iba1[0]):(xmax+ offset_Iba1[0])]
    #print(xmin)
    #print(xmin + offset_NeuN[0])
    if (subset_NeuN3 > 0).all() and subset_NeuN3.size != 0:
        if 1==0: # Gaussian mode # Ctr5mのときのみぼやかす。 
            #ここに、subset_NeuN3をx,yがsigma2, zが2/(2.5/0.65) のgaussianをかけてそれをrankdataに渡すことにする
            subset_NeuN3 = median_filter(subset_NeuN3, size = 5)#sigma=[sigma_z, sigma_xy, sigma_xy])
        if 1==1:
            # ガウシアンフィルターを適用
            subset_NeuN3 = gaussian_filter(subset_NeuN3, sigma=[2.0, 0, 0])

        ranks_NeuN = rankdata(-subset_NeuN3, method='average').reshape(
            subset_NeuN3.shape) 
        rank_NeuN_average = np.average(ranks_NeuN [zmin_s:zmax_s, ymin_s:ymax_s, xmin_s:xmax_s])
        
        del ranks_NeuN, subset_NeuN3
        
        # それぞれのzスライスに対して平均値を計算
        # average_per_slice = [
        #    np.average(ranks_NeuN[z, ymin_s:ymax_s, xmin_s:xmax_s])
        #    for z in range(zmin_s, zmax_s)
        #]

        if 1==0:
            # ranks_NeuN の10%の要素をシャッフル
            total_elements = ranks_NeuN.size
            num_elements_to_shuffle = int(total_elements * 0.5)  # 50%の要素数
            indices_to_shuffle = np.random.choice(total_elements, num_elements_to_shuffle, replace=False)
            elements_to_shuffle = ranks_NeuN.flatten()[indices_to_shuffle]
            np.random.shuffle(elements_to_shuffle)
            ranks_NeuN.flatten()[indices_to_shuffle] = elements_to_shuffle
            # zスライスの平均値の中で最小のものを選ぶ rank 最小は最も相対intensityが高いということ
            rank_NeuN_average = np.average(ranks_NeuN [zmin_s:zmax_s, ymin_s:ymax_s, xmin_s:xmax_s]) #min(average_per_slice)
            del ranks_NeuN, subset_NeuN3

        if 1==0:
            # ranks_NeuN の指定された範囲の50%の要素を選択
            selected_region = ranks_NeuN[zmin_s:zmax_s, ymin_s:ymax_s, xmin_s:xmax_s]
            total_elements_region = selected_region.size
            num_elements_to_shuffle_region = int(total_elements_region * 0.5)  # 選択された範囲の50%
            indices_to_shuffle_region = np.random.choice(total_elements_region, num_elements_to_shuffle_region, replace=False)
            elements_to_shuffle_region = selected_region.flatten()[indices_to_shuffle_region]

            # ranks_NeuN 全体から同じ数の要素を選択
            total_elements = ranks_NeuN.size
            indices_to_shuffle_total = np.random.choice(total_elements, num_elements_to_shuffle_region, replace=False)
            elements_to_shuffle_total = ranks_NeuN.flatten()[indices_to_shuffle_total]

            # 両方の要素グループをシャッフルし、それぞれの位置に挿入
            np.random.shuffle(elements_to_shuffle_region)
            np.random.shuffle(elements_to_shuffle_total)
            selected_region.flatten()[indices_to_shuffle_region] = elements_to_shuffle_total
            ranks_NeuN.flatten()[indices_to_shuffle_total] = elements_to_shuffle_region

            # zスライスの平均値を計算
            rank_NeuN_average = np.average(selected_region)  # 修正された範囲の平均を計算
            del ranks_NeuN, subset_NeuN3 

        if 1==0:
            selected_region = ranks_NeuN[zmin_s:zmax_s, ymin_s:ymax_s, xmin_s:xmax_s]
            # selected_region 以外の領域のインデックスを取得
            outside_indices = np.array(np.nonzero(~np.isin(ranks_NeuN, selected_region)))

            # selected_region から50%の要素をランダムに選択
            total_elements_region = selected_region.size
            num_elements_to_shuffle_region = int(total_elements_region * 0.7)
            indices_to_shuffle_region = np.random.choice(total_elements_region, num_elements_to_shuffle_region, replace=False)

            # outside_indices から同数の要素をランダムに選択
            indices_to_shuffle_outside = np.random.choice(outside_indices.shape[1], num_elements_to_shuffle_region, replace=False)
            selected_outside_elements = outside_indices[:, indices_to_shuffle_outside]

            # 互いに置換
            selected_region_flat = selected_region.flatten()
            selected_region_flat[indices_to_shuffle_region], ranks_NeuN[tuple(selected_outside_elements)] = \
                ranks_NeuN[tuple(selected_outside_elements)], selected_region_flat[indices_to_shuffle_region]

            # 元の形状に戻す
            selected_region[:] = selected_region_flat.reshape(selected_region.shape)

            # zスライスの平均値を計算
            rank_NeuN_average = np.average(selected_region)  # 修正された範囲の平均を計算
            del ranks_NeuN, subset_NeuN3 

        if 1 == 0:
            selected_region = ranks_NeuN[zmin_s:zmax_s, ymin_s:ymax_s, xmin_s:xmax_s]

            # selected_region 内のランク値が下位50%の要素を特定
            threshold_rank = np.percentile(selected_region, 70)  # 下位50%の閾値
            lower_half_indices = np.where(selected_region <= threshold_rank) # 閾値以下にした。つまり、ランクが高め(良い目)のものにfocus　→　今度は悪い目のものにfocus

            # 全体平均のランク値を計算
            total_volume = (2*half_ROI_size_xy + 1)*(2*half_ROI_size_xy + 1)*(2*half_ROI_size_z + 1)
            average_rank_value = total_volume * (3 / 4) #total_volume / 2

            # 下位50%の要素を平均ランク値で置換
            selected_region[lower_half_indices] = average_rank_value

            # zスライスの平均値を再計算
            rank_NeuN_average = np.average(selected_region) - total_volume * (1 / 3)
            del ranks_NeuN, subset_NeuN3


    if (subset_Iba13 > 0).all() and subset_Iba13.size != 0:

        if 1==1:
            # ガウシアンフィルターを適用
            subset_Iba13 = gaussian_filter(subset_Iba13, sigma=[2.0, 0, 0])
    
        ranks_Iba1 = rankdata(-subset_Iba13, method='average').reshape(
            subset_Iba13.shape)
        # それぞれのzスライスに対して平均値を計算
        #average_per_slice = [
        #    np.average(ranks_Iba1[z, ymin_s:ymax_s, xmin_s:xmax_s])
        #    for z in range(zmin_s, zmax_s)
        #]

        # zスライスの平均値の中で最小のものを選ぶ
        rank_Iba1_average = np.average(ranks_Iba1 [zmin_s:zmax_s, ymin_s:ymax_s, xmin_s:xmax_s]) #min(average_per_slice)
        del ranks_Iba1, subset_Iba13

    return np.array([rank_NeuN_average, rank_Iba1_average], dtype=np.float32)



def main():
    args = docopt(__doc__)

    with open(args["PARAM_FILE"]) as f:
        params_multichannel = json.load(f)
    wbc = WholeBrainCells(params_multichannel["paramfile_nucl"])
    #wbc = WholeBrainImages(params_multichannel["paramfile_nucl"])

    clf_manual3 = joblib.load(params_multichannel["clf_file"])

    left_stack_xmin = int(params_multichannel["left_stack_xmin"])
    offset_right_NeuN = params_multichannel["offset_right_NeuN"]#int(params_multichannel["offset_right_NeuN"])
    offset_right_Iba1 = params_multichannel["offset_right_Iba1"]#int(params_multichannel["offset_right_Iba1"])
    offset_left_NeuN = params_multichannel["offset_left_NeuN"]#int(params_multichannel["offset_left_NeuN"])
    offset_left_Iba1 = params_multichannel["offset_left_Iba1"]#int(params_multichannel["offset_left_Iba1"])

    # print("offset_right_NeuN " + str(offset_right_NeuN))

    result_dir_FW = os.path.join(
        wbc.halfbrain_cells_FW.halfbrain_images.params["dst_basedir"])
    result_dir_RV = os.path.join(
        wbc.halfbrain_cells_RV.halfbrain_images.params["dst_basedir"])
    if not os.path.exists(os.path.join(params_multichannel["dst_basedir"], "FW")):
        os.makedirs(os.path.join(params_multichannel["dst_basedir"], "FW"))
    if not os.path.exists(os.path.join(params_multichannel["dst_basedir"], "RV")):
        os.makedirs(os.path.join(params_multichannel["dst_basedir"], "RV"))

    joblib.Parallel(n_jobs=int(args["-p"]), verbose=10)([
        joblib.delayed(save_intensities)(
            xyname=(xname, yname),
            resultfile_nucl=os.path.join(
                result_dir_FW, "{}_{}.bin".format(yname, xname)),
            imagestack_nucl=wbc.halfbrain_cells_FW.dict_stacks[(
                xname, yname)].imagestack,
            # imagestack_second = wbc2.halfbrain_cells_FW.dict_stacks[(xname,yname)].imagestack,
            clf=clf_manual3,
            dst_basedir=os.path.join(params_multichannel["dst_basedir"], "FW"),
            left_stack_xmin=left_stack_xmin,
            offset_right_NeuN=offset_right_NeuN,
            offset_right_Iba1=offset_right_Iba1,
            offset_left_NeuN=offset_left_NeuN,
            offset_left_Iba1=offset_left_Iba1,
            FWRV="FW"
        )
        for (xname, yname) in wbc.halfbrain_cells_FW.dict_stacks.keys()
        if os.path.exists(os.path.join(result_dir_FW, "{}_{}.bin".format(yname, xname)))
    ] + [
        joblib.delayed(save_intensities)(
            xyname=(xname, yname),
            resultfile_nucl=os.path.join(
                result_dir_RV, "{}_{}.bin".format(yname, xname)),
            imagestack_nucl=wbc.halfbrain_cells_RV.dict_stacks[(
                xname, yname)].imagestack,
            # imagestack_second = wbc2.halfbrain_cells_RV.dict_stacks[(xname,yname)].imagestack,
            clf=clf_manual3,
            dst_basedir=os.path.join(params_multichannel["dst_basedir"], "RV"),
            left_stack_xmin=left_stack_xmin,
            offset_right_NeuN=offset_right_NeuN,
            offset_right_Iba1=offset_right_Iba1,
            offset_left_NeuN=offset_left_NeuN,
            offset_left_Iba1=offset_left_Iba1,
            FWRV="RV"
        )
        for (xname, yname) in wbc.halfbrain_cells_RV.dict_stacks.keys()
        if os.path.exists(os.path.join(result_dir_RV, "{}_{}.bin".format(yname, xname)))
    ])


if __name__ == "__main__":
    main()
