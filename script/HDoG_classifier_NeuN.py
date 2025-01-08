#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Overview:
  Classify cell candidates and determine true cells

Usage:
  HDoG_classifier.py PARAM_FILE

Options:
  -h --help        Show this screen.
  --version        Show version.
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.ndimage
import os.path, joblib
import tifffile
import json
from docopt import docopt

from sklearn.mixture import BayesianGaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from sklearn.pipeline import Pipeline
import graphviz
import gc
from joblib import Parallel, delayed

from MergeBrain_NeuN import WholeBrainCells

dt_classified = np.dtype([
    ('is_positive','bool'), ('intensity','f4'),('intensity_2','f4'), ('intensity_3','f4')
])

dt_scalemerged = np.dtype([
    ('scaled_x','f4'), ('scaled_y', 'f4'), ('scaled_z', 'f4'),
    ('is_valid', 'bool'),
])

class ThresholdClassifier:
    """
    Simple classifier with threshold on first feature.
    the data with first feature larger than the specified threshold is regarded as positive.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, X):
        return X[:,0] > self.threshold


class LinearClassifier2D:
    """
    Sample satisfies `a * X[:,0] + b * X[:,1] + c > 0` is regarded as positive.
    """
    def __init__(self, a,b,c):
        self.a = a
        self.b = b
        self.c = c

    def predict(self, X):
        return self.a * X[:,0] + self.b * X[:,1] + self.c > 0


def process_src_pkl(src_pkl, wbc, normalizer_img, dt_scalemerged, **kwargs):
    yname,xname = src_pkl.split("/")[-1].split(".")[0].split("_")
    if "FW" in src_pkl:
        cellstack = wbc.halfbrain_cells_FW.get_stack_by_xyname(xname=xname, yname=yname)
    else:
        cellstack = wbc.halfbrain_cells_RV.get_stack_by_xyname(xname=xname, yname=yname)

    #print(src_pkl)
    data_scalemerged_stack = np.fromfile(src_pkl, dtype = dt_scalemerged)
    X = get_X_with_normalizer(cellstack.data_local, data_scalemerged_stack, normalizer_img, is_ave=wbc.is_ave, **kwargs)

    del cellstack, data_scalemerged_stack
    gc.collect()

    return X

def process_src_pkl_2(src_pkl, wbc, normalizer_img, dt_scalemerged, **kwargs):
    yname,xname = src_pkl.split("/")[-1].split(".")[0].split("_")

    if "FW" in src_pkl:
        cellstack = wbc.halfbrain_cells_FW.get_stack_by_xyname(xname=xname, yname=yname)
    else:
        cellstack = wbc.halfbrain_cells_RV.get_stack_by_xyname(xname=xname, yname=yname)

    #print(src_pkl)
    data_scalemerged_stack = np.fromfile(src_pkl, dtype = dt_scalemerged)
    #X = get_X_with_normalizer(cellstack.data_local, data_scalemerged_stack, normalizer_img, is_ave=wbc.is_ave, **kwargs)
    X = get_X_3d(cellstack.data_local)

    del cellstack, data_scalemerged_stack
    gc.collect()

    return X

def get_X_whole(wbc, normalizer_img=None, **kwargs):
    info = np.fromfile(os.path.join(wbc.wholebrain_images.params["dst_basedir"], "info.bin"), dtype=[('directory', 'U400'), ('num', 'i4')])
    
    n_jobs = 20 # 使用するコア数。-1 はすべてのコアを使用することを意味します。
    X_whole = Parallel(n_jobs=n_jobs, backend='threading')(delayed(process_src_pkl)(src_pkl, wbc, normalizer_img, dt_scalemerged, **kwargs) for src_pkl in info['directory'])

    X_whole = np.concatenate(X_whole)
    return X_whole

def process_data_scalemerged(src_pkl, wbc, dt_scalemerged, clf=None, normalizer_img=None, return_X=False, **kwargs):
    yname,xname = src_pkl.split("/")[-1].split(".")[0].split("_")
    if "FW" in src_pkl:
        cellstack = wbc.halfbrain_cells_FW.get_stack_by_xyname(xname=xname, yname=yname)
    else:
        cellstack = wbc.halfbrain_cells_RV.get_stack_by_xyname(xname=xname, yname=yname)
    
    data_scalemerged_stack = np.fromfile(src_pkl, dtype = dt_scalemerged)
    
    results = {}
    
    if clf:
        X = get_X_with_normalizer(cellstack.data_local, data_scalemerged_stack, normalizer_img, is_ave=wbc.is_ave, **kwargs)
        if X.shape[0] == 0:
            return None
        pred = clf.predict(X)
        results['data'] = data_scalemerged_stack[data_scalemerged_stack["is_valid"]][pred]
        if return_X:
            results['X'] = X
    else:
        results['data'] = data_scalemerged_stack[data_scalemerged_stack["is_valid"]]
        if return_X:
            results['X'] = get_X_with_normalizer(cellstack.data_local, data_scalemerged_stack, normalizer_img, is_ave=wbc.is_ave, **kwargs)
    
    del cellstack, data_scalemerged_stack
    gc.collect()


    return results

def get_data_scalemerged_whole(wbc, clf=None, normalizer_img=None, return_X=False, **kwargs):
    info = np.fromfile(os.path.join(wbc.wholebrain_images.params["dst_basedir"], "info.bin"), dtype=[('directory', 'U400'), ('num', 'i4')])
    
    n_jobs = 20 # すべてのコアを使用
    results = Parallel(n_jobs=n_jobs, backend='threading')(delayed(process_data_scalemerged)(src_pkl, wbc, dt_scalemerged, clf, normalizer_img, return_X, **kwargs) for src_pkl in info['directory'])
    
    # filter out None results
    results = [res for res in results if res is not None]
    
    data_scalemerged_whole = np.concatenate([res['data'] for res in results])
    
    if not return_X:
        del results
        gc.collect()
        return data_scalemerged_whole
    else:
        X_whole = np.concatenate([res['X'] for res in results if 'X' in res])
        del results
        gc.collect()
        return data_scalemerged_whole, X_whole

def make_density_image(wbc, clf=None, dtype=np.uint16, normalizer_img=None, **kwargs):
    data_scalemerged_whole = get_data_scalemerged_whole(wbc, clf, normalizer_img, **kwargs)

    depth = int(np.floor(np.max(data_scalemerged_whole["scaled_z"])))
    height = int(np.floor(np.max(data_scalemerged_whole["scaled_y"])))
    width = int(np.floor(np.max(data_scalemerged_whole["scaled_x"])))
    density_img,_ = np.histogramdd(
        np.vstack([
            data_scalemerged_whole["scaled_z"],
            data_scalemerged_whole["scaled_y"],
            data_scalemerged_whole["scaled_x"]
        ]).T,
        bins=(depth, height, width),
        range=[(0,depth-1),(0,height-1),(0,width-1)]
    )
    return density_img.astype(dtype)


# Feature Vector
def get_X_with_normalizer(data_local, data_scalemerged, normalizer_img=None, **kwargs):

    data_local_valid = data_local[data_scalemerged["is_valid"]]

    if data_local_valid.shape[0] == 0:
        #return np.empty((0,2), dtype=np.float32) 
        return np.empty((0,6), dtype=np.float32)  #new

    X = get_X_3d(data_local_valid, **kwargs)

    if normalizer_img is not None:
        data_scalemerged_valid = data_scalemerged[data_scalemerged["is_valid"]]
        X[:, 0] -= np.log10(normalizer_img[
            np.clip(np.floor(data_scalemerged_valid["scaled_z"]).astype(int),
                    a_min=0,a_max=normalizer_img.shape[0]-1),
            np.clip(np.floor(data_scalemerged_valid["scaled_y"]).astype(int),
                    a_min=0,a_max=normalizer_img.shape[1]-1),
            np.clip(np.floor(data_scalemerged_valid["scaled_x"]).astype(int),
                    a_min=0,a_max=normalizer_img.shape[2]-1)
        ])
        #del data_scalemerged_valid
        #gc.collect()

    del data_local_valid
    gc.collect()

    return X

def get_X_3d(data,
             bias=np.array([0, 0, 0]),
             scale=np.array([1.0, 1.0, 1.0]),
             bias_before_log=np.array([0.0,0.0,0.0]),
             is_ave=False): # false
    if not is_ave:
        _X = np.array([
            data["intensity"],
            data["structureness"],
            data["blobness"]
        ]).T
    else:
        _X = np.array([
            data["intensity"] / data["size"],
            data["structureness"],
            data["blobness"]
        ]).T

    # X1の生成
    X1 = np.array([data['intensity'], data['intensity2'], data['intensity3']]).T 
    X1 = np.nan_to_num(X1)

    # X0の生成
    #X0 = np.nan_to_num((np.log10((_X + bias_before_log), where=((_X + bias_before_log) != 0)) + bias) * scale)

    X0 = np.log10(np.where(_X == 0.0, 0.00001, _X))
    # X0とX1の結合
    X = np.concatenate([X0, X1], 1)

    del X0, X1
    gc.collect()

    return X


def predict_unsupervised(X, i_feature_maximize=1, n_components=3, **vargs):
    vbgmm = BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process",
        n_components=n_components,
        **vargs
    )
    vbgmm.fit(X)
    pred = vbgmm.predict(X)

    if i_feature_maximize is None:
        return pred
    else:
        i_pred_cluster = np.argmax(vbgmm.means_[:,i_feature_maximize])
        return pred == i_pred_cluster

def train_decision_tree(X, y, max_depth=2):
    pipe = Pipeline([
        ('pca', PCA()),
        ('tree', DecisionTreeClassifier(max_depth=max_depth))
    ])
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=.2, shuffle=True)
    pipe.fit(X_train, y_train)
    print("Validation Score:{}".format(pipe.score(X_test, y_test)))
    pred = pipe.predict(X)
    return pipe, pred

def show_decision_tree(pipe, feature_names=["intensity","structureness","blobness"]):
    if isinstance(pipe, Pipeline):
        clf = pipe.named_steps['tree']
    elif isinstance(pipe, DecisionTreeClassifier):
        clf = pipe
    dot_data = export_graphviz(clf, out_file=None,
                               feature_names=feature_names,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    return graph

def plot_classification_result(feature_x, feature_y, pred):
    sns.scatterplot(feature_x, feature_y, hue=pred)
    #if np.count_nonzero(pred == 0) > 1:
    sns.kdeplot(feature_x[pred==0],feature_y[pred==0], cmap="Blues", shade=True, shade_lowest=False)
    #if np.count_nonzero(pred == 1) > 1:
    sns.kdeplot(feature_x[pred==1],feature_y[pred==1], cmap="Oranges", shade=True, shade_lowest=False)
    return

def plot_features_for_each_stacks(r, func_get_X, clf=None):
    i_xs = []
    i_ys = []
    x1s = []
    x2s = []
    preds=[]
    for i_y in range(len(r.stack_ys)):
        print(i_y)
        for i_x in range(len(r.stack_xs)):
            _stack = r.get_stack(i_xy=(i_x,i_y), verbose=False)
            if len(_stack.df) < 10000: continue
            _X_stack = func_get_X(_stack.df)
            if clf: _pred_stack = clf.predict(_X_stack)
            for i in range(0,len(_stack.df), 1000):
                i_xs.append(i_x)
                i_ys.append(i_y)
                x1s.append(_X_stack[i,0])
                x2s.append(_X_stack[i,1])
                if clf: preds.append(_pred_stack[i])

    df_plot = pd.DataFrame({
        "i_x":pd.Series(i_xs),
        "i_y":pd.Series(i_ys),
        "X1":pd.Series(x1s),
        "X2":pd.Series(x2s),
        "predicted":pd.Series(preds)
    })
    if not clf:
        g = sns.FacetGrid(df_plot, row="i_y", col="i_x",
                          height=2, aspect=1,)
        g.map(sns.kdeplot, "X1", "X2", shade=True, shade_lowest=False)
    else:
        g = sns.FacetGrid(df_plot, row="i_y", col="i_x",hue="predicted",
                      hue_kws={"cmap":["Blues", "Oranges"]},
                      height=2, aspect=1,)
        g.map(sns.kdeplot, "X1", "X2",)

def make_pred_img(df, pred, zlim=None, sigma=(1.0, 2.0, 2.0)):
    # zlim = (start_z, end_z) or None(:=all)
    # sigma = (sigma_z, sigma_y, sigma_x) or None(:=load from parameter file)
    if not zlim:
        z0 = 0
        depth = int(np.max(np.array(df["local_x"])))
    else:
        z0 = zlim[0]
        depth = zlim[1]-zlim[0]
    if not sigma:
        raise ValueError

    pred_img = np.zeros((depth,2048,2048), dtype=np.uint16)
    for i in np.nonzero(pred)[0]:
        z = int(np.array(df["local_z"])[i] - z0)
        if z < 0 or z >= depth: continue
        y = int(np.array(df["local_y"])[i])
        x = int(np.array(df["local_x"])[i])
        #pred_img[z-1:z+2, y-1:y+2, x-1:x+2] = 1000
        pred_img[z, y, x] = 5000
    pred_img = scipy.ndimage.filters.gaussian_filter(pred_img, sigma=sigma, truncate=2.)
    return pred_img

def save_classified_result_for_stack(dst_path, cellstack, clf):
    X = get_X_3d(cellstack.data_local)

    if X.shape[0] == 0:
        data_clf = np.zeros(0, dtype=dt_classified)
    else:
        data_clf = np.zeros(X.shape[0], dtype=dt_classified)
        data_clf["is_positive"] = clf.predict(X)
        data_clf["intensity"] =  X[:,3]
        data_clf["intensity_2"] =  X[:,4] #> 0.2
        data_clf["intensity_3"] =  X[:,5] #> 0.2
    #joblib.dump(data_clf, dst_path, compress=3)
    print(dst_path)
    data_clf.tofile(dst_path)
    #del X
    #gc.collect()
    return np.count_nonzero(data_clf["is_positive"])


def save_classified_result_for_stack_2(dst_path, X, clf):
    #X = get_X_3d(cellstack.data_local)
    
    if X.shape[0] == 0:
        data_clf = np.zeros(0, dtype=dt_classified)
    else:
        data_clf = np.zeros(X.shape[0], dtype=dt_classified)
        data_clf["is_positive"] = clf.predict(X)
        data_clf["intensity"] =  X[:,3]
        data_clf["intensity_2"] =  X[:,4] #> 0.2
        data_clf["intensity_3"] =  X[:,5] #> 0.2
    #joblib.dump(data_clf, dst_path, compress=3)
    print(dst_path)
    data_clf.tofile(dst_path)
    #del X
    #gc.collect()
    return np.count_nonzero(data_clf["is_positive"])


def main():
    args = docopt(__doc__)

    with open(args["PARAM_FILE"]) as f:
        params = json.load(f)

    wbc = WholeBrainCells(params["MergeBrain_paramfile"])
    dst_basedir = params["dst_basedir"]
    dst_basedir_FW = os.path.join(dst_basedir,"FW")
    dst_basedir_RV = os.path.join(dst_basedir,"RV")
    if not os.path.exists(dst_basedir_FW):
        os.makedirs(dst_basedir_FW)
    if not os.path.exists(dst_basedir_RV):
        os.makedirs(dst_basedir_RV)

    # prepare feature vectors
    print("[*] preparing feature vectors...")
    # X_whole = get_X_whole(wbc)
    #print("Total candidates:", X_whole.shape[0])
    num_skip = params["num_skip_samples"]

    num_components = params["automatic_boundary"]["num_clusters"]
    i_feature_maximize = params["automatic_boundary"]["i_feature_maximize"]

    #print("[*] running unsupervised clustering...")
    #pred_un = predict_unsupervised(
    #    X_whole[::num_skip,:],
    #    n_components=num_components,
    #    i_feature_maximize=i_feature_maximize
    #)

    #if params["use_manual_boundary"]:
    #    print("[*] creating manual classifier...")
    #    #a = int(params["manual_boundary"]["a"])
    #    #b = int(params["manual_boundary"]["b"])
    #    #c = int(params["manual_boundary"]["c"])
    a = params["manual_boundary"]["a"]
    b = params["manual_boundary"]["b"]
    c = params["manual_boundary"]["c"]
    clf = LinearClassifier2D(a,b,c)
    #pred = clf.predict(X_whole[::num_skip])

    #else:
    #    # train supervised decision tree classifier
    #    print("[*] training supervised classifier...")
    #    clf, pred = train_decision_tree(X_whole[::num_skip], pred_un)
    
    print(c)

    # save classifier
    clf_filename = os.path.join(
        params["dst_basedir"],
        "{}.pkl".format(params["classifier_name"]))
    joblib.dump(clf, clf_filename)
    #clf.tofile(clf_filename)
    print("[*] saved the classifier to {}".format(clf_filename))

    # save classified results & count total number of cells
    print("[*] saving classified results...")

    def classify_and_save_for_stack(src_pkl, wbc, clf, normalizer_img, dt_scalemerged, **kwargs):
        # データを読み込む
        X = process_src_pkl_2 (src_pkl, wbc, normalizer_img, dt_scalemerged, **kwargs)
        
        # 分類器を使って予測を行う
        #pred = clf.predict(X)
        
        # 結果を保存する
        yname, xname = src_pkl.split("/")[-1].split(".")[0].split("_")
        if "FW" in src_pkl:
            dst_path = os.path.join(dst_basedir_FW, "{}_{}.bin".format(yname, xname))
        else:
            dst_path = os.path.join(dst_basedir_RV, "{}_{}.bin".format(yname, xname))
        num_cell = save_classified_result_for_stack_2(dst_path, X, clf)
        
        return num_cell

    info = np.fromfile(os.path.join(wbc.wholebrain_images.params["dst_basedir"], "info.bin"), dtype=[('directory', 'U400'), ('num', 'i4')])
    dict_num_cells = {}
    #for xyname,cellstack in wbc.halfbrain_cells_FW.dict_stacks.items():
    if 1==0:
        for src_pkl in info['directory']:
            #if cellstack.is_dummy: continue
            yname, xname = src_pkl.split("/")[-1].split(".")[0].split("_")
            dst_path = os.path.join(dst_basedir_FW, "{}_{}.bin".format(yname,xname))
            #num_cells = save_classified_result_for_stack(dst_path, cellstack, clf)
            num_cells = classify_and_save_for_stack(src_pkl, wbc, clf, None, dt_scalemerged)
            dict_num_cells[dst_path] = num_cells

    # 並列処理のための関数
    def process_and_save_cells(src_pkl):
        yname, xname = src_pkl.split("/")[-1].split(".")[0].split("_")
        dst_path = os.path.join(dst_basedir_FW, "{}_{}.bin".format(yname,xname))
        num_cells = classify_and_save_for_stack(src_pkl, wbc, clf, None, dt_scalemerged)
        return dst_path, num_cells
        
    n_jobs = 20  # 使用するコア数。-1はすべてのコアを使用することを意味します。
    results = Parallel(n_jobs=n_jobs, backend='threading')(delayed(process_and_save_cells)(src_pkl) for src_pkl in info['directory'])

    dict_num_cells = dict(results)  # 結果を辞書に変換

    #for xyname,cellstack in wbc.halfbrain_cells_RV.dict_stacks.items():
    #    if cellstack.is_dummy: continue
    #    dst_path = os.path.join(dst_basedir_RV, "{}_{}.bin".format(xyname[1],xyname[0]))
    #    #num_cells = save_classified_result_for_stack(dst_path, cellstack, clf)
    #    num_cells = classify_and_save_for_stack(src_pkl, wbc, clf, None, dt_scalemerged)
    #    dict_num_cells[dst_path] = num_cells

    # save statistics
    #joblib.dump(dict_num_cells, os.path.join(dst_basedir, "info.pkl"), compress=3)
    data_to_save = np.array([(k, v) for k, v in dict_num_cells.items()], dtype=[('directory', 'U400'), ('num', 'i4')])
    data_to_save.tofile(os.path.join(dst_basedir, "info.bin"))

    total_cells = sum(list(dict_num_cells.values()))
    print("Total cells:", total_cells)
    #print("NeuN+ cells:", num_cells2)
    #print("Iba1+ cells:", num_cells3)


    if 1==0:
        # make cell density image
        print("[*] making cell density image...")
        density_img = make_density_image(wbc, clf)
        img_filename = os.path.join(
            params["dst_basedir"],
            "{}.tif".format(params["density_img_name"]))
        tifffile.imsave(
            img_filename,
            density_img.astype(np.uint16)
        )

        #New: imge_density.tif save to sclemerged folder
        #with open(params["MergeBrain_paramfile"]) as f2:
        #    params_m = json.load(f2)
            
        #img_filename_m = os.path.join(
        #params_m["dst_basedir"],
        #    "{}.tif".format(params["density_img_name"]))
        img_filename_m = img_filename.replace("classified_", "scalemerged_").replace("img_density", "img_density_initial")
        tifffile.imsave(
            img_filename_m,
            density_img.astype(np.uint16)
        )
        print("[*] saved density image to {}".format(img_filename))
        print("[*] saved density image to {}".format(img_filename_m))

    if 1==0:
        fig = plt.figure(figsize=(10,5))
        fig.add_subplot(1,2,1)
        plot_classification_result(
            X_whole[::num_skip,0],
            X_whole[::num_skip,1],
            pred_un
        )
        plt.title("unsupervised clustering result")
        fig.add_subplot(1,2,2)
        plot_classification_result(
            X_whole[::num_skip,0],
            X_whole[::num_skip,1],
            pred
        )
        if params["use_manual_boundary"]:
            plt.title("manual classification result")
        else:
            plt.title("supervised classification result")
        plt.savefig(os.path.join(
            params["dst_basedir"],
            "feature_space_{}.png".format(params["classifier_name"])
        ))

    return

if __name__ == "__main__":
    main()
