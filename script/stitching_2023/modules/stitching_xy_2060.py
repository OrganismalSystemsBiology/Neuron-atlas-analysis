from modules.utils_220828 import *
import gc
dt_local = np.dtype([
    ('local_x', 'f4'), ('local_y', 'f4'), ('local_z', 'f4'),
    ('structureness', 'f4'), ('blobness', 'f4'), ('intensity', 'f4'),
    ('size', 'u2'), ('padding', 'u2'), ('intensity2', 'f4'), ('intensity3', 'f4')
])

dt_stitched = np.dtype([
    ('stitched_x', 'f4'), ('stitched_y', 'f4'), ('stitched_z', 'f4')
])


class stitching_xy():
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
        self.x_step_original = self.x_pixel-2*self.x_overlap
        self.y_step_original = self.y_pixel-2*self.y_overlap
        self.FW_boundary = self.param['merge_info']["boundary_fname"]["FW_boundary"]
        self.RV_boundary = self.param['merge_info']["boundary_fname"]["RV_boundary"]

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
        self.scalemerged_dst_path = self.param['dst_scalemerged_basedir']
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

        # x_y stitchingの画像の大きさ
        # core_short
        self.short_search = self.param["xy_stitching"]["short_search"]
        self.long_search = self.param["xy_stitching"]["long_search"]
        self.xy_stitching_z_search = self.param["xy_stitching"]["z_search"]

        # Left_target
        self.Left_target_ymin = 0
        self.Left_target_ymax = self.y_pixel
        self.Left_target_xmin = self.x_pixel - 2*self.x_overlap
        self.Left_target_xmax = self.x_pixel

        # right_core
        self.Right_core_ymin = self.long_search
        self.Right_core_ymax = self.y_pixel - self.long_search
        self.Right_core_xmin = self.short_search
        self.Right_core_xmax = 2*self.x_overlap - self.short_search

        # up_target
        self.Up_target_ymin = self.y_pixel - 2*self.y_overlap
        self.Up_target_ymax = self.y_pixel
        self.Up_target_xmin = 0
        self.Up_target_xmax = self.x_pixel

        # down_core
        self.Down_core_ymin = self.short_search
        self.Down_core_ymax = 2*self.y_overlap - self.short_search
        self.Down_core_xmin = self.long_search
        self.Down_core_xmax = self.x_pixel - self.long_search

        # FW_RV_stitching
        self.rasen_check_range = 25
        self.FW_RV_xy_scan_range = 100  # xy_scan_range(100*2+1pixel scanする)
        self.FW_RV_z_scan_range = 100  # z_scan_range(100pixel scanする)
        # must be int*zstep
        self.FW_RV_stitching_scale = self.param["FW_RV_stitching"]["stitching_scale"]
        self.z_step = int(self.FW_RV_stitching_scale/self.scale_z)
        self.FW_ants_maisuu = 5
        self.RV_ants_maisuu = 151
        self.large_transform_path = self.stitching_dst_path+"/large_transform.mat"
        self.RV_maisuu_path = self.stitching_dst_path+"/RV_maisuu.txt"
        self.RV_large_moved_center_path = self.stitching_dst_path + \
            "/check/RV_large_moved_center.tif"
        make_dir(os.path.join(self.stitching_dst_path, "stitched/")) #new
        make_dir(os.path.join(self.stitching_dst_path, "stitched/FW")) #new
        make_dir(os.path.join(self.stitching_dst_path, "stitched/RV")) #new
        make_dir(os.path.join(self.stitching_dst_path, "xy_stitching_result/"))
        make_dir(os.path.join(self.stitching_dst_path, "FW_RV_stitching_result/"))
        make_dir(os.path.join(self.stitching_check_path, "RV_merge_boundary"))
        make_dir(os.path.join(self.stitching_check_path, "FW_merge_boundary"))
        make_dir(os.path.join(self.stitching_check_path, "FW_done"))
        make_dir(os.path.join(self.stitching_check_path, "RV_done"))
        make_dir(os.path.join(self.stitching_check_path, "FW_target"))
        make_dir(os.path.join(self.stitching_check_path, "RV_moving"))
        make_dir(os.path.join(self.stitching_check_path, "RV_moved"))
        make_dir(os.path.join(self.stitching_check_path, "FW_RVresult"))
        make_dir(os.path.join(self.stitching_check_path, "FW_RVtarget"))
        self.check_FW_large_path = os.path.join(
            self.stitching_check_path, "check_FW_large")
        self.check_RV_large_path = os.path.join(
            self.stitching_check_path, "check_RV_large")
        make_dir(self.check_FW_large_path)
        make_dir(self.check_RV_large_path)

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
        for i in range(100):
            rasen_x = rasen_x + list(range(-1*i, i+1))
            rasen_x = rasen_x + [i+1]*(i+1)*2
            rasen_x = rasen_x + list(range(i, -1*(i+1), -1))
            rasen_x = rasen_x + [-1*(i+1)]*(i+1)*2
        rasen_y = [0]
        for i in range(100):
            rasen_y = rasen_y + list(range(-1*i, i+1))
            rasen_y = rasen_y + [i+1]*((i+1)*2+1)
            rasen_y = rasen_y + list(range(i, -1*(i+1), -1))
            rasen_y = rasen_y + [-1*(i+1)]*((i+1)*2+1)
        rasen_xy = []
        # 1000tileまで対応
        for i in range(1000):
            rasen_xy.append([rasen_y[i], rasen_x[i]])
        rasen_xy = np.array(
            rasen_xy)+np.array([int(self.x_list_num/2), int(self.y_list_num/2)])
        self.rasen_xy = rasen_xy[(rasen_xy.T[0] < self.x_list_num) * (
            rasen_xy.T[1] < self.y_list_num) * (rasen_xy.T[0] >= 0) * (rasen_xy.T[1] >= 0)]
        self.center_stack_x, self.center_stack_y = rasen_xy[0]

        # self.FW_z_maxはFWのstackの中での最大のZ,self.RV_z_maxはRVのstackの中での最大のZ
        file_name = []
        for curDir, dirs, files in os.walk(self.src_base_dir_FW):
            for file in files:
                if file.endswith(".bin"):
                    file_name.append(int(os.path.splitext(file)[0]))
        if len(file_name) == 0:
            self.FW_z_max = 0
        if len(file_name) != 0:
            self.FW_z_max = max(file_name)

        file_name = []
        for curDir, dirs, files in os.walk(self.src_base_dir_RV):
            for file in files:
                if file.endswith(".bin"):
                    file_name.append(int(os.path.splitext(file)[0]))
        if len(file_name) == 0:
            self.RV_z_max = 0
        if len(file_name) != 0:
            self.RV_z_max = max(file_name)

    ###################### 基本的動作###############################
    def import_bin(self, src_bin_path, FWorRV, _dtype=np.uint16):
        if os.path.exists(src_bin_path):
            # print("src_bin_path is {}".format(src_bin_path))
            img = np.fromfile(src_bin_path, dtype=_dtype)
            if len(img) != 2048*2060:
                img = np.zeros((2048, 2048))
            else:
                img = img.reshape(2048, 2060)
                img = img[:2048, :2048]
                
                # new hot pixel対応
                hot_pixels = [(1871, 1557), (1627, 1716), (1401, 1730), (946, 843), (946, 841), (900, 1573), (812, 184)]
                for x, y in hot_pixels:
                    img[x, y] = img[x-1:x+2, y-1:y+2].mean()

        if not os.path.exists(src_bin_path):
            # print("no data {}".format(src_bin_path))
            img = np.zeros((2048, 2048))
        return img

    # def import_bin(self, src_bin_path, FWorRV, _dtype=np.uint16):
    #   if os.path.exists(src_bin_path):
    #     #print("src_bin_path is {}".format(src_bin_path))
    #     img = np.fromfile(src_bin_path, dtype=_dtype)
    #     if len(img) != 2048*2048:
    #       img = np.zeros((2048,2048))
    #     else:
    #       img = img.reshape(2048, 2048)
    #   if not os.path.exists(src_bin_path):
    #     #print("no data {}".format(src_bin_path))
    #     img = np.zeros((2048,2048))
    #   return img

    def import_img_xyz(self, x_num, y_num, z_num, FWorRV):
        if FWorRV == "FW":
            xx, yy = self.list_x_FW[x_num], self.list_y_FW[y_num]
            z_name = str(z_num).zfill(5)
            y_path_list = sorted(os.listdir(self.src_base_dir_FW))
            x_path_list = sorted(os.listdir(os.path.join(
                self.src_base_dir_FW, y_path_list[yy])))
            stack_path = self.src_base_dir_FW + "/" + \
                y_path_list[yy] + "/" + x_path_list[xx]
        if FWorRV == "RV":
            xx, yy = self.list_x_RV[x_num], self.list_y_RV[y_num]
            z_name = str(z_num).zfill(5)
            y_path_list = sorted(os.listdir(self.src_base_dir_RV))
            x_path_list = sorted(os.listdir(os.path.join(
                self.src_base_dir_RV, y_path_list[yy])))
            stack_path = self.src_base_dir_RV + "/" + \
                y_path_list[yy] + "/" + x_path_list[xx]
        src_image_path_list = glob.glob(
            os.path.join(stack_path, "*{}.bin".format(z_name)))
        if len(src_image_path_list) == 0:
            src_image_path = "no_file"
        if len(src_image_path_list) != 0:
            src_image_path = src_image_path_list[0]
        # print(src_image_path)
        img = self.import_bin(src_image_path, FWorRV)
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
            size=50, mode="reflect", cval=0)
        erosion_l_img = scipy.ndimage.filters.uniform_filter(
            scipy.ndimage.morphology.grey_erosion(
                src_img, size=50, mode="reflect").astype(np.float32),
            size=50, mode="reflect", cval=0)

        intensity = src_img.astype(np.float32)
        sabun_img = dilation_l_img - erosion_l_img
        sabun_img[sabun_img == 0] = 1
        norm_img = (intensity >= self.bcg_intensity) * \
            (intensity - erosion_l_img) / sabun_img
        return norm_img

    def norm_large(self, x_num, y_num, z_num, FWorRV):
        src_img = self.import_img_xyz(x_num, y_num, z_num, FWorRV)
        dilation_l_img = scipy.ndimage.filters.uniform_filter(
            scipy.ndimage.morphology.grey_dilation(
                src_img, size=600, mode="reflect").astype(np.float32),
            size=600, mode="reflect", cval=0)
        erosion_l_img = scipy.ndimage.filters.uniform_filter(
            scipy.ndimage.morphology.grey_erosion(
                src_img, size=600, mode="reflect").astype(np.float32),
            size=600, mode="reflect", cval=0)

        intensity = src_img.astype(np.float32)
        sabun_img = dilation_l_img - erosion_l_img
        sabun_img[sabun_img == 0] = 1
        norm_img = (intensity >= self.bcg_intensity) * \
            (intensity - erosion_l_img) / sabun_img
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

    def check_img_ZZ(self, micro, ZZ, FWorRV):
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
                a = self.import_img_xyz(XX, YY, ZZ, FWorRV)
                b = cv2.resize(a, dsize=None, fx=self.scale_x /
                               micro, fy=self.scale_y/micro)
                down_y_stack, down_x_stack = np.shape(b)
                # bb =np.split(np.array(np.split(a,int(self.x_pixel/downscale_ratio),1)),int(self.y_pixel/downscale_ratio),1)
                img[y_step*YY:y_step*YY+down_y_stack, x_step *
                    XX:x_step*XX+down_x_stack] = b*(b > 0)
        if FWorRV == "FW":
            save_img_path = os.path.join(
                self.check_FW_large_path, "{0}_{1}_check_before.tif".format(FWorRV, ZZ))
        if FWorRV == "RV":
            save_img_path = os.path.join(
                self.check_RV_large_path, "{0}_{1}_check_before.tif".format(FWorRV, ZZ))
        tifffile.imsave(save_img_path, img.astype(np.uint16))
        return
    ##################### x-y-stitching##############################

    def compare_to_left(self, XX, YY, ZZ, FWorRV):
        z_search_range = self.xy_stitching_z_search
        z_search_maisuu = 2*z_search_range + 1
        Left_img = self.norm(XX-1, YY, ZZ, FWorRV)
        Left_target = Left_img[self.Left_target_ymin:self.Left_target_ymax,
                               self.Left_target_xmin: self.Left_target_xmax]
        Right_core_images = [self.norm(XX, YY, ZZ+z_search-z_search_range, FWorRV)[self.Right_core_ymin:self.Right_core_ymax,
                                                                                   self.Right_core_xmin:self.Right_core_xmax] for z_search in range(z_search_maisuu)]
        R = []
        for i, template_img in enumerate(Right_core_images):
            # 画像の検索（Template Matching）
            result = cv2.matchTemplate(
                Left_target, template_img, cv2.TM_CCORR_NORMED)
            # 検索結果の信頼度と位置座標の取得
            R.append(cv2.minMaxLoc(result))
        Sz = (np.argmax(np.array(R).T[1]))
        Sx, Sy = R[Sz][3]
        z_diff, y_diff, x_diff = Sz-z_search_range, Sy - \
            self.long_search, Sx-self.short_search
        return np.array([x_diff, y_diff, z_diff, np.max(np.array(R).T[1])])

    def compare_to_up(self, XX, YY, ZZ, FWorRV):
        z_search_range = self.xy_stitching_z_search
        z_search_maisuu = 2*z_search_range + 1
        Up_img = self.norm(XX, YY-1, ZZ, FWorRV)
        Up_target = Up_img[self.Up_target_ymin:self.Up_target_ymax,
                           self.Up_target_xmin:self.Up_target_xmax]
        Down_core_images = [self.norm(XX, YY, ZZ+z_search-z_search_range, FWorRV)[self.Down_core_ymin:self.Down_core_ymax,
                                                                                  self.Down_core_xmin:self.Down_core_xmax] for z_search in range(z_search_maisuu)]
        R = []
        for i, template_img in enumerate(Down_core_images):
            # 画像の検索（Template Matching）
            result = cv2.matchTemplate(
                Up_target, template_img, cv2.TM_CCORR_NORMED)
            # 検索結果の信頼度と位置座標の取得
            R.append(cv2.minMaxLoc(result))
        Sz = (np.argmax(np.array(R).T[1]))
        Sx, Sy = R[Sz][3]
        z_diff, y_diff, x_diff = Sz-z_search_range, Sy - \
            self.short_search, Sx-self.long_search
        return np.array([x_diff, y_diff, z_diff, np.max(np.array(R).T[1])])

    def compare_left_and_up(self, stack_num, ZZ, FWorRV):
        YY, XX = divmod(stack_num, self.x_list_num)
        print(FWorRV, XX, YY, ZZ)
        z_search_range = self.xy_stitching_z_search
        z_search_maisuu = 2*z_search_range + 1
        image_stitchable = self.stitchable(XX, YY, ZZ, FWorRV)
        if image_stitchable["left"] == 0 and image_stitchable["up"] == 0:
            Result_left = np.array([np.nan, np.nan, np.nan, np.nan])
            Result_up = np.array([np.nan, np.nan, np.nan, np.nan])
        else:
            core_images = np.array([self.norm(
                XX, YY, ZZ + z_search - z_search_range, FWorRV) for z_search in range(z_search_maisuu)])
            Right_core_images = core_images[:, self.Right_core_ymin:self.Right_core_ymax,
                                            self.Right_core_xmin:self.Right_core_xmax]
            Down_core_images = core_images[:, self.Down_core_ymin:self.Down_core_ymax,
                                           self.Down_core_xmin:self.Down_core_xmax]
            del core_images
            Left_img = self.norm(XX-1, YY, ZZ, FWorRV)
            Left_target = Left_img[self.Left_target_ymin:self.Left_target_ymax,
                                   self.Left_target_xmin: self.Left_target_xmax]
            Up_img = self.norm(XX, YY-1, ZZ, FWorRV)
            Up_target = Up_img[self.Up_target_ymin:self.Up_target_ymax,
                               self.Up_target_xmin:self.Up_target_xmax]

            if image_stitchable["left"] == 0:
                Result_left = np.array([np.nan, np.nan, np.nan, np.nan])
            if image_stitchable["left"] != 0:
                R_left = []
                for i, template_img in enumerate(Right_core_images):
                    # 画像の検索（Template Matching）
                    result = cv2.matchTemplate(
                        Left_target, template_img, cv2.TM_CCORR_NORMED)
                    # 検索結果の信頼度と位置座標の取得
                    R_left.append(cv2.minMaxLoc(result))
                Sz = (np.argmax(np.array(R_left).T[1]))
                Sx, Sy = R_left[Sz][3]
                z_diff, y_diff, x_diff = Sz-z_search_range, Sy - \
                    self.long_search, Sx-self.short_search
                Result_left = np.array(
                    [x_diff, y_diff, z_diff, np.max(np.array(R_left).T[1])])

            if image_stitchable["up"] == 0:
                Result_up = np.array([np.nan, np.nan, np.nan, np.nan])
            if image_stitchable["up"] != 0:
                R_up = []
                for i, template_img in enumerate(Down_core_images):
                    # 画像の検索（Template Matching）
                    result = cv2.matchTemplate(
                        Up_target, template_img, cv2.TM_CCORR_NORMED)
                    # 検索結果の信頼度と位置座標の取得
                    R_up.append(cv2.minMaxLoc(result))
                Sz = (np.argmax(np.array(R_up).T[1]))
                Sx, Sy = R_up[Sz][3]
                z_diff, y_diff, x_diff = Sz-z_search_range, Sy - \
                    self.short_search, Sx-self.long_search
                Result_up = np.array(
                    [x_diff, y_diff, z_diff, np.max(np.array(R_up).T[1])])

        save_left_path = os.path.join(os.path.join(self.stitching_dst_path, "xy_stitching_result/"),
                                      FWorRV+"_X"+str(XX).zfill(5)+"_Y"+str(YY).zfill(5)+"_Z"+str(ZZ).zfill(5)+"_left")
        save_up_path = os.path.join(os.path.join(self.stitching_dst_path, "xy_stitching_result/"),
                                    FWorRV+"_X"+str(XX).zfill(5)+"_Y"+str(YY).zfill(5)+"_Z"+str(ZZ).zfill(5)+"_up")
        np.save(save_left_path, Result_left)
        np.save(save_up_path, Result_up)
        return

    # 16*2048の画像が張り合わせられるか判断する関数(tate_yoko == 0 tate,tate_yoko = 1 yoko)
    def obi(self, obi_img, tate_yoko):
        block = np.array(np.split(obi_img, 128, tate_yoko))
        # 端の64pixelはサンプルがあってもstitchできない判定にする[4:-4]
        block_mean = block.mean(axis=(1, 2))[4:-4]
        # print(len(block_mean))
        block_zero = np.array(block_mean > self.bcg_intensity)
        block_mean_filter = block_zero + \
            np.append(0, np.delete(block_zero, -1)) + \
            np.append(np.delete(block_zero, 0), 0)  # dilation_filter
        block_mean_filter = block_mean_filter + \
            np.append(0, np.append(0, np.delete(
                np.delete(block_zero, -1), -1)))
        block_mean_filter = block_mean_filter + \
            np.append(np.append(np.delete(np.delete(block_zero, 0), 0), 0), 0)
        if np.sum(block_mean_filter > 4.5) > 0.5:
            return 1
        if np.sum(block_mean_filter > 4.5) < 0.5:
            return 0

    def stitchable(self, XX, YY, z_num, FWorRV):
        src_img = self.import_img_xyz(XX, YY, z_num, FWorRV)
        top_img = src_img[:16, :]
        bottom_img = src_img[-16:, :]
        left_img = src_img[:, :16]
        right_img = src_img[:, -16:]
        return {"up": self.obi(top_img, 1), "down": self.obi(bottom_img, 1), "left": self.obi(left_img, 0), "right": self.obi(right_img, 0)}

    def z_list_xy(self, XX, YY, FWorRV):
        if FWorRV == "FW":
            list_x = self.list_x_FW
            list_y = self.list_y_FW
            y_name_list = sorted(os.listdir(self.src_base_dir_FW))
            x_name_list = sorted(os.listdir(os.path.join(
                self.src_base_dir_FW, y_name_list[list_y[YY]])))
            z_list = sorted(os.listdir(self.src_base_dir_FW + "/" +
                            y_name_list[list_y[YY]] + "/" + x_name_list[list_x[XX]] + "/"))
        if FWorRV == "RV":
            list_x = self.list_x_RV
            list_y = self.list_y_RV
            y_name_list = sorted(os.listdir(self.src_base_dir_RV))
            x_name_list = sorted(os.listdir(os.path.join(
                self.src_base_dir_RV, y_name_list[list_y[YY]])))
            z_list = sorted(os.listdir(self.src_base_dir_RV + "/" +
                            y_name_list[list_y[YY]] + "/" + x_name_list[list_x[XX]] + "/"))
        return z_list

    def xy_stitcing_plane_parallel(self, ZZZ, FWorRV):
        XY_stitching_ = joblib.Parallel(n_jobs=45)(joblib.delayed(self.compare_left_and_up)(StackNum, ZZZ, FWorRV)
                                                   for StackNum, ZZZ, FWorRV
                                                   in zip(range(self.x_list_num*self.y_list_num), [ZZZ]*self.x_list_num*self.y_list_num, [FWorRV]*self.x_list_num*self.y_list_num))
        return

    def Stitching_Data_XY(self, XX, YY, FWorRV, Left_UP):
        if XX == 0 and Left_UP == "left":
            StitchingDataXY = np.array([np.nan, np.nan, np.nan, np.nan])
        else:
            if YY == 0 and Left_UP == "up":
                StitchingDataXY = np.array([np.nan, np.nan, np.nan, np.nan])
            else:
                if XX == self.x_list_num and Left_UP == "left":
                    StitchingDataXY = np.array(
                        [np.nan, np.nan, np.nan, np.nan])
                else:
                    if YY == self.y_list_num and Left_UP == "up":
                        StitchingDataXY = np.array(
                            [np.nan, np.nan, np.nan, np.nan])
                    else:
                        StitchingDataXY = np.array([np.load(stitching_file)
                                                   for stitching_file
                                                   in glob.glob(os.path.join(os.path.join(
                                                       self.stitching_dst_path, "xy_stitching_result/"),
                                                       FWorRV+"_X"+str(XX).zfill(5)+"_Y"+str(YY).zfill(5)+"*"+"_{}.npy".format(Left_UP)))])
                        StitchingDataXY = StitchingDataXY[StitchingDataXY.T[3] > 0.9]
                        if len(StitchingDataXY) == 0:
                            StitchingDataXY = np.array(
                                [np.nan, np.nan, np.nan, np.nan])
                        else:
                            StitchingDataXY = np.median(
                                StitchingDataXY, axis=0).astype(np.int16)
        return StitchingDataXY

    def Stitching_Data_XY_all(self, XX, YY, FWorRV, Left_UP):
        if Left_UP == "left" or Left_UP == "up":
            return self.Stitching_Data_XY(XX, YY, FWorRV, Left_UP)[:3]
        if Left_UP == "right":
            if XX == self.x_list_num - 1:
                return np.array([np.nan, np.nan, np.nan, np.nan])[:3]
            if XX != self.x_list_num - 1:
                return -self.Stitching_Data_XY(XX+1, YY, FWorRV, "left")[:3]
        if Left_UP == "down":
            if YY == self.y_list_num - 1:
                return np.array([np.nan, np.nan, np.nan, np.nan])[:3]
            if YY != self.y_list_num - 1:
                return -self.Stitching_Data_XY(XX, YY+1, FWorRV, "up")[:3]

    # 隣にサンプルがあるかどうか(それぞれの境界で)
    def ari_or_nasi(self, XX, YY, FWorRV):  # LEFT,RIHT,UP.DOWN
        ari_nasi = np.zeros(4)  # LEFT,RIHT,UP.DOWN
        if not np.isnan(self.Stitching_Data_XY(XX, YY, FWorRV, "left")[0]):
            ari_nasi[0] = 1
        if not np.isnan(self.Stitching_Data_XY(XX+1, YY, FWorRV, "left")[0]):
            ari_nasi[1] = 1
        if not np.isnan(self.Stitching_Data_XY(XX, YY, FWorRV, "up")[0]):
            ari_nasi[2] = 1
        if not np.isnan(self.Stitching_Data_XY(XX, YY+1, FWorRV, "up")[0]):
            ari_nasi[3] = 1
        return ari_nasi

    def clc_shift_FW_and_shift_RV(self):
        COUNTER = 0
        for xxx, yyy in self.rasen_xy:
            if np.sum(self.ari_or_nasi(xxx, yyy, "FW"))*np.sum(self.ari_or_nasi(xxx, yyy, "RV")) > 0:
                COUNTER = COUNTER + 1
                first_fill = xxx, yyy
                break
        if COUNTER == 0:
            print("ERROR: There is no stitched stack in RASEN range!!!")
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)

        ##### FW ######
        ut1 = time.time()
        print("making shift_FW......")
        shift_FW = np.zeros(
            (self.x_list_num, self.y_list_num, 3)).astype(np.int32)
        done_FW = np.zeros((self.y_list_num, self.x_list_num))
        # 「あり」を一個うめる
        done_FW[first_fill[1], first_fill[0]] = 1
        shift_FW[first_fill[0], first_fill[1]] = np.array(
            [self.x_step_original*(first_fill[0]+1), self.y_step_original*(first_fill[1]+1), 0])
        filled_number_before, filled_number_after = 0, 1
        while (filled_number_after < self.x_list_num*self.y_list_num):
            # らせん状に埋めていく
            while (filled_number_before < filled_number_after):
                filled_number_before = np.sum(done_FW)
                next_FW = cv2.dilate(done_FW, kernel, iterations=1)-done_FW
                next_FW_index = np.array(np.where(next_FW != 0)).T
                tifffile.imsave(os.path.join(self.stitching_check_path, "FW_done/{}.tif".format(
                    str(int(filled_number_before)).zfill(5))), done_FW.astype(np.uint16))
                for yy, xx in next_FW_index:
                    counter = 0
                    # print(xx,yy)
                    tmp = np.zeros(3)
                    if not np.isnan(self.Stitching_Data_XY_all(xx, yy, "FW", "up")[0]):
                        if done_FW[yy-1, xx] == 1:
                            counter += 1
                            tmp = tmp + shift_FW[xx, yy-1] + np.array(
                                [0, self.y_step_original, 0]) + self.Stitching_Data_XY_all(xx, yy, "FW", "up")
                    if not np.isnan(self.Stitching_Data_XY_all(xx, yy, "FW", "left")[0]):
                        if done_FW[yy, xx-1] == 1:
                            counter += 1
                            tmp = tmp + shift_FW[xx-1, yy] + np.array(
                                [self.x_step_original, 0, 0]) + self.Stitching_Data_XY_all(xx, yy, "FW", "left")
                    if not np.isnan(self.Stitching_Data_XY_all(xx, yy, "FW", "down")[0]):
                        if done_FW[yy+1, xx] == 1:
                            counter += 1
                            tmp = tmp + shift_FW[xx, yy+1] + np.array(
                                [0, -self.y_step_original, 0]) + self.Stitching_Data_XY_all(xx, yy, "FW", "down")
                    if not np.isnan(self.Stitching_Data_XY_all(xx, yy, "FW", "right")[0]):
                        if done_FW[yy, xx+1] == 1:
                            counter += 1
                            tmp = tmp + shift_FW[xx+1, yy] + np.array(
                                [-self.x_step_original, 0, 0]) + self.Stitching_Data_XY_all(xx, yy, "FW", "right")
                    if counter == 0:
                        # 繋がらなかったら飛ばす
                        continue
                    shift_FW[xx, yy] = tmp/counter
                    done_FW[yy, xx] = 1
                filled_number_after = np.sum(done_FW)
            tifffile.imsave(os.path.join(self.stitching_check_path, "FW_done/{}.tif".format(str(int(
                filled_number_before)).zfill(5))), np.zeros((self.y_list_num, self.x_list_num)).astype(np.uint16))
            if not filled_number_after == self.x_list_num*self.y_list_num:  # 繋がっていないstackが無ければ終了
                filled_number_before, filled_number_after = np.sum(
                    done_FW), np.sum(done_FW)+1
                # 周りのなにもないところも埋めていく
                flag = 0  # 繋がっているstackが出てくればwhileから抜けるためのflag
                while (filled_number_before < filled_number_after and flag == 0):
                    tifffile.imsave(os.path.join(self.stitching_check_path, "FW_done/{}.tif".format(
                        str(int(filled_number_before)).zfill(5))), done_FW.astype(np.uint16))
                    filled_number_before = np.sum(done_FW)
                    next_FW = cv2.dilate(done_FW, kernel, iterations=1)-done_FW
                    next_FW_index = np.array(np.where(next_FW != 0)).T
                    for yy, xx in next_FW_index:
                        if np.sum(self.ari_or_nasi(xx, yy, "FW")) == 0:
                            if yy > 0 and done_FW[yy-1, xx] == 1:
                                shift_FW[xx, yy] = shift_FW[xx, yy-1] + \
                                    np.array([0, self.y_step_original, 0])
                            else:
                                if xx > 0 and done_FW[yy, xx-1] == 1:
                                    shift_FW[xx, yy] = shift_FW[xx-1, yy] + \
                                        np.array([self.x_step_original, 0, 0])
                                else:
                                    if yy < self.y_list_num-1 and done_FW[yy+1, xx] == 1:
                                        shift_FW[xx, yy] = shift_FW[xx, yy+1] + \
                                            np.array(
                                                [0, -self.y_step_original, 0])
                                    else:
                                        if xx < self.x_list_num-1 and done_FW[yy, xx+1] == 1:
                                            shift_FW[xx, yy] = shift_FW[xx+1, yy] + \
                                                np.array(
                                                    [-self.x_step_original, 0, 0])
                            done_FW[yy, xx] = 1
                        else:
                            if yy > 0 and done_FW[yy-1, xx] == 1:
                                shift_FW[xx, yy] = shift_FW[xx, yy-1] + \
                                    np.array([0, self.y_step_original, 0])
                            else:
                                if xx > 0 and done_FW[yy, xx-1] == 1:
                                    shift_FW[xx, yy] = shift_FW[xx-1, yy] + \
                                        np.array([self.x_step_original, 0, 0])
                                else:
                                    if yy < self.y_list_num-1 and done_FW[yy+1, xx] == 1:
                                        shift_FW[xx, yy] = shift_FW[xx, yy+1] + \
                                            np.array(
                                                [0, -self.y_step_original, 0])
                                    else:
                                        if xx < self.x_list_num-1 and done_FW[yy, xx+1] == 1:
                                            shift_FW[xx, yy] = shift_FW[xx+1, yy] + \
                                                np.array(
                                                    [-self.x_step_original, 0, 0])
                            done_FW[yy, xx] = 1
                            flag = 1
                            break
                    filled_number_after = np.sum(done_FW)
        # zのゼロ点をcenter_stackのzにする
        self.shift_FW = (
            shift_FW - np.array([0, 0, shift_FW[self.center_stack_x, self.center_stack_y][2]])).astype(np.int64)
        print("making shift_FW takes {}s".format(str(time.time()-ut1)))
        np.save(os.path.join(self.stitching_check_path,
                "shift_FW.npy"), self.shift_FW)

        ##### RV ######
        ut1 = time.time()
        print("making shift_RV.......")
        shift_RV = np.zeros(
            (self.x_list_num, self.y_list_num, 3)).astype(np.int32)
        done_RV = np.zeros((self.y_list_num, self.x_list_num))

        # 「あり」を一個うめる
        done_RV[first_fill[1], first_fill[0]] = 1
        shift_RV[first_fill[0], first_fill[1]] = np.array(
            [self.x_step_original*(first_fill[0]+1), self.y_step_original*(first_fill[1]+1), 0])
        filled_number_before, filled_number_after = 0, 1
        while (filled_number_after < self.x_list_num*self.y_list_num):
            # らせん状に埋めていく
            while (filled_number_before < filled_number_after):
                filled_number_before = np.sum(done_RV)
                next_RV = cv2.dilate(done_RV, kernel, iterations=1)-done_RV
                next_RV_index = np.array(np.where(next_RV != 0)).T
                tifffile.imsave(os.path.join(self.stitching_check_path, "RV_done/{}.tif".format(
                    str(int(filled_number_before)).zfill(5))), done_RV.astype(np.uint16))
                for yy, xx in next_RV_index:
                    counter = 0
                    # print(xx,yy)
                    tmp = np.zeros(3)
                    if not np.isnan(self.Stitching_Data_XY_all(xx, yy, "RV", "up")[0]):
                        if done_RV[yy-1, xx] == 1:
                            counter += 1
                            tmp = tmp + shift_RV[xx, yy-1] + np.array(
                                [0, self.y_step_original, 0]) + self.Stitching_Data_XY_all(xx, yy, "RV", "up")
                    if not np.isnan(self.Stitching_Data_XY_all(xx, yy, "RV", "left")[0]):
                        if done_RV[yy, xx-1] == 1:
                            counter += 1
                            tmp = tmp + shift_RV[xx-1, yy] + np.array(
                                [self.x_step_original, 0, 0]) + self.Stitching_Data_XY_all(xx, yy, "RV", "left")
                    if not np.isnan(self.Stitching_Data_XY_all(xx, yy, "RV", "down")[0]):
                        if done_RV[yy+1, xx] == 1:
                            counter += 1
                            tmp = tmp + shift_RV[xx, yy+1] + np.array(
                                [0, -self.y_step_original, 0]) + self.Stitching_Data_XY_all(xx, yy, "RV", "down")
                    if not np.isnan(self.Stitching_Data_XY_all(xx, yy, "RV", "right")[0]):
                        if done_RV[yy, xx+1] == 1:
                            counter += 1
                            tmp = tmp + shift_RV[xx+1, yy] + np.array(
                                [-self.x_step_original, 0, 0]) + self.Stitching_Data_XY_all(xx, yy, "RV", "right")
                    if counter == 0:
                        # 繋がらなかったら飛ばす
                        continue
                    shift_RV[xx, yy] = tmp/counter
                    done_RV[yy, xx] = 1
                filled_number_after = np.sum(done_RV)

            tifffile.imsave(os.path.join(self.stitching_check_path, "RV_done/{}.tif".format(str(int(
                filled_number_before)).zfill(5))), np.zeros((self.y_list_num, self.x_list_num)).astype(np.uint16))
            if not filled_number_after == self.x_list_num*self.y_list_num:  # 繋がっていないstackが無ければ終了
                filled_number_before, filled_number_after = np.sum(
                    done_RV), np.sum(done_RV)+1
                # 周りのなにもないところも埋めていく
                flag = 0  # 繋がっているstackが出てくればwhileから抜けるためのflag
                while (filled_number_before < filled_number_after and flag == 0):
                    tifffile.imsave(os.path.join(self.stitching_check_path, "RV_done/{}.tif".format(
                        str(int(filled_number_before)).zfill(5))), done_RV.astype(np.uint16))
                    filled_number_before = np.sum(done_RV)
                    next_RV = cv2.dilate(done_RV, kernel, iterations=1)-done_RV
                    next_RV_index = np.array(np.where(next_RV != 0)).T
                    tifffile.imsave(os.path.join(self.stitching_check_path, "RV_done/{}.tif".format(
                        str(int(np.sum(done_RV))).zfill(5))), done_RV.astype(np.uint16))
                    for yy, xx in next_RV_index:
                        if np.sum(self.ari_or_nasi(xx, yy, "RV")) == 0:
                            if yy > 0 and done_RV[yy-1, xx] == 1:
                                shift_RV[xx, yy] = shift_RV[xx, yy-1] + \
                                    np.array([0, self.y_step_original, 0])
                            else:
                                if xx > 0 and done_RV[yy, xx-1] == 1:
                                    shift_RV[xx, yy] = shift_RV[xx-1, yy] + \
                                        np.array([self.x_step_original, 0, 0])
                                else:
                                    if yy < self.y_list_num-1 and done_RV[yy+1, xx] == 1:
                                        shift_RV[xx, yy] = shift_RV[xx, yy+1] + \
                                            np.array(
                                                [0, -self.y_step_original, 0])
                                    else:
                                        if xx < self.x_list_num-1 and done_RV[yy, xx+1] == 1:
                                            shift_RV[xx, yy] = shift_RV[xx+1, yy] + \
                                                np.array(
                                                    [-self.x_step_original, 0, 0])
                            done_RV[yy, xx] = 1
                        tifffile.imsave(os.path.join(self.stitching_check_path, "RV_done/{}.tif".format(str(int(
                            filled_number_before)).zfill(5))), 2*np.ones((self.y_list_num, self.x_list_num)).astype(np.uint16))
                        if np.sum(self.ari_or_nasi(xx, yy, "RV")) != 0:
                            if yy > 0 and done_RV[yy-1, xx] == 1:
                                shift_RV[xx, yy] = shift_RV[xx, yy-1] + \
                                    np.array([0, self.y_step_original, 0])
                            else:
                                if xx > 0 and done_RV[yy, xx-1] == 1:
                                    shift_RV[xx, yy] = shift_RV[xx-1, yy] + \
                                        np.array([self.x_step_original, 0, 0])
                                else:
                                    if yy < self.y_list_num-1 and done_RV[yy+1, xx] == 1:
                                        shift_RV[xx, yy] = shift_RV[xx, yy+1] + \
                                            np.array(
                                                [0, -self.y_step_original, 0])
                                    else:
                                        if xx < self.x_list_num-1 and done_RV[yy, xx+1] == 1:
                                            shift_RV[xx, yy] = shift_RV[xx+1, yy] + \
                                                np.array(
                                                    [-self.x_step_original, 0, 0])
                            done_RV[yy, xx] = 1
                            flag = 1
                            tifffile.imsave(os.path.join(self.stitching_check_path, "RV_done/{}.tif".format(
                                str(int(np.sum(done_RV))).zfill(5))), done_RV.astype(np.uint16))
                            tifffile.imsave(os.path.join(self.stitching_check_path, "RV_done/{}.tif".format(str(int(
                                filled_number_before)).zfill(5))), np.ones((self.y_list_num, self.x_list_num)).astype(np.uint16))
                            break
                    filled_number_after = np.sum(done_RV)
        # zのゼロ点をcenter_stackのzにする
        self.shift_RV = (
            shift_RV - np.array([0, 0, shift_RV[self.center_stack_x, self.center_stack_y][2]])).astype(np.int64)
        print("making shift_RV takes {}s".format(str(time.time()-ut1)))
        np.save(os.path.join(self.stitching_check_path,
                "shift_RV.npy"), self.shift_RV)

    def check_stitch_RV_boundary(self, ZZ):
        micro = 1
        downscale_ratio = micro/self.scale_x
        shift_RV = self.shift_RV
        shift_FWorRV = (
            shift_RV/np.array([downscale_ratio, downscale_ratio, 1])).astype(np.int64)
        down_y = int(self.y_pixel*(self.y_list_num+1)/downscale_ratio)
        down_x = int(self.x_pixel*(self.x_list_num+1)/downscale_ratio)
        img = np.zeros((down_y, down_x))
        y_overlap_half, x_overlap_half = int(
            self.y_overlap/2), int(self.x_overlap/2)
        for YY in range(self.y_list_num):
            for XX in range(self.x_list_num):
                a = self.import_img_xyz(XX, YY, ZZ+shift_FWorRV[XX, YY, 2], "RV")[
                    y_overlap_half:-y_overlap_half, x_overlap_half:-x_overlap_half]
                b = cv2.resize(a, dsize=None, fx=self.scale_x /
                               micro, fy=self.scale_y/micro)
                down_y_stack, down_x_stack = np.shape(b)
                # bb =np.split(np.array(np.split(a,int(self.x_pixel/downscale_ratio),1)),int(self.y_pixel/downscale_ratio),1)
                y_start = shift_FWorRV[XX, YY, 1] + \
                    int(y_overlap_half*self.scale_y/micro)
                x_start = shift_FWorRV[XX, YY, 0] + \
                    int(x_overlap_half*self.scale_x/micro)
                img[y_start:y_start+down_y_stack,
                    x_start:x_start+down_x_stack] = b
        save_img_path = os.path.join(
            self.stitching_check_path, "RV_merge_boundary/{}_check_stitch.tif".format(str(ZZ).zfill(8)))
        tifffile.imsave(save_img_path, img.astype(np.uint16))
        return

    def check_stitch_FW_boundary(self, ZZ):
        micro = 1
        downscale_ratio = micro/self.scale_x
        shift_FW = self.shift_FW
        shift_FWorRV = (
            shift_FW/np.array([downscale_ratio, downscale_ratio, 1])).astype(np.int64)
        down_y = int(self.y_pixel*(self.y_list_num+1)/downscale_ratio)
        down_x = int(self.x_pixel*(self.x_list_num+1)/downscale_ratio)
        img = np.zeros((down_y, down_x))
        y_overlap_half, x_overlap_half = int(
            self.y_overlap/2), int(self.x_overlap/2)
        for YY in range(self.y_list_num):
            for XX in range(self.x_list_num):
                a = self.import_img_xyz(XX, YY, ZZ+shift_FWorRV[XX, YY, 2], "FW")[
                    y_overlap_half:-y_overlap_half, x_overlap_half:-x_overlap_half]
                b = cv2.resize(a, dsize=None, fx=self.scale_x /
                               micro, fy=self.scale_y/micro)
                down_y_stack, down_x_stack = np.shape(b)
                # bb =np.split(np.array(np.split(a,int(self.x_pixel/downscale_ratio),1)),int(self.y_pixel/downscale_ratio),1)
                y_start = shift_FWorRV[XX, YY, 1] + \
                    int(y_overlap_half*self.scale_y/micro)
                x_start = shift_FWorRV[XX, YY, 0] + \
                    int(x_overlap_half*self.scale_x/micro)
                img[y_start:y_start+down_y_stack,
                    x_start:x_start+down_x_stack] = b
        save_img_path = os.path.join(
            self.stitching_check_path, "FW_merge_boundary/{}_check_stitch.tif".format(str(ZZ).zfill(8)))
        tifffile.imsave(save_img_path, img.astype(np.uint16))
        return


class stitching_basic_functions(stitching_xy):
    def __init__(self, param, stitching_img_um):
        super().__init__(param, stitching_img_um)

    # XX,YY(global stack番号)入れるとstack名（y_x）を返す関数
    def get_stack_name(self, x_num, y_num, FWorRV):
        if FWorRV == "FW":
            x_num = self.list_x_FW[x_num]
            y_num = self.list_y_FW[y_num]
            y_path_list = sorted(os.listdir(self.src_base_dir_FW))
            x_path_list = sorted(os.listdir(os.path.join(
                self.src_base_dir_FW, y_path_list[y_num])))
            return x_path_list[x_num]  # y_x
        if FWorRV == "RV":
            x_num = self.list_x_RV[x_num]
            y_num = self.list_y_RV[y_num]
            y_path_list = sorted(os.listdir(self.src_base_dir_RV))
            x_path_list = sorted(os.listdir(os.path.join(
                self.src_base_dir_RV, y_path_list[y_num])))
            return x_path_list[x_num]  # y_x
    # stitching後の座標をstackごとにsave。lenはcandidate_nuと同じ、つまり各stackの全部にする。
    def save_stitched_stack_data(self, x_num, y_num, FWorRV, stack_data_stitched):
        stack_name = self.get_stack_name(x_num, y_num, FWorRV)
        #os.path.join(self.stitching_dst_path, "stitched/")
        if FWorRV == "FW":
            HDoG_nu_data = self.param_nu_FW["dst_basedir"]+stack_name+".bin"
            dst_data = os.path.join(self.stitching_dst_path, "stitched/FW/")+stack_name+".bin"
            #もしcandidate_nuのfileが存在すれば、保存する
            if os.path.exists(HDoG_nu_data):
                print(stack_data_stitched.shape)
                tmp_stack_data_stitched = np.zeros(shape=stack_data_stitched.shape[1], dtype=dt_stitched)
                print(tmp_stack_data_stitched.shape)
                tmp_stack_data_stitched["stitched_x"] = stack_data_stitched [0]
                tmp_stack_data_stitched["stitched_y"] = stack_data_stitched [1]
                tmp_stack_data_stitched["stitched_z"] = stack_data_stitched [2]
                #joblib.dump(tmp_stack_data_stitched, dst_data, compress=3)
                tmp_stack_data_stitched.tofile(dst_data)

        if FWorRV == "RV":
            HDoG_nu_data = self.param_nu_FW["dst_basedir"]+stack_name+".bin"
            dst_data = os.path.join(self.stitching_dst_path, "stitched/RV/")+stack_name+".bin"
            #もしcandidate_nuのfileが存在すれば、保存する
            if os.path.exists(HDoG_nu_data):
                tmp_stack_data_stitched = np.zeros(shape=stack_data_stitched.shape[1], dtype=dt_stitched)
                tmp_stack_data_stitched["stitched_x"] = stack_data_stitched [0]
                tmp_stack_data_stitched["stitched_y"] = stack_data_stitched [1]
                tmp_stack_data_stitched["stitched_z"] = stack_data_stitched [2]
                #joblib.dump(tmp_stack_data_stitched, dst_data, compress=3)
                tmp_stack_data_stitched.tofile(dst_data)
        return

    # nuclearのHDoG結果よみこみ
    def load_nu_HDoG_data(self, x_num, y_num, FWorRV):
        stack_name = self.get_stack_name(x_num, y_num, FWorRV)
        if FWorRV == "FW":
            HDoG_nu_data = self.param_nu_FW["dst_basedir"]+stack_name+".bin"
            if not os.path.exists(HDoG_nu_data):
                return np.array([], dtype=dt_local)#[]
            data_local_nu = np.fromfile(HDoG_nu_data, dtype=dt_local)
        if FWorRV == "RV":
            HDoG_nu_data = self.param_nu_RV["dst_basedir"]+stack_name+".bin"
            if not os.path.exists(HDoG_nu_data):
                return np.array([], dtype=dt_local) #[]
            data_local_nu = np.fromfile(HDoG_nu_data, dtype=dt_local)
        return data_local_nu

    # nuclearのclassify結果の読み込み
    def load_nu_classify_data(self, x_num, y_num, FWorRV):
        stack_name = self.get_stack_name(x_num, y_num, FWorRV)
        if FWorRV == "FW":
            HDoG_nu_data = self.param_nu_FW["dst_basedir"]+stack_name+".bin"
            if not os.path.exists(HDoG_nu_data):
                return []
        if FWorRV == "RV":
            HDoG_nu_data = self.param_nu_RV["dst_basedir"]+stack_name+".bin"
            if not os.path.exists(HDoG_nu_data):
                return []
        clf_data_path = os.path.join(
            self.param_clf["dst_basedir"], FWorRV+"/"+stack_name+".pkl")
        clf_data = joblib.load(clf_data_path)
        return clf_data['is_positive']

    def load_point_data(self, xx, yy, FWorRV):
        nu_classify_data = self.load_nu_classify_data(xx, yy, FWorRV)
        if len(nu_classify_data) == 0:
            return np.array([])
        nu_HDoG_data = self.load_nu_HDoG_data(xx, yy, FWorRV)[nu_classify_data]

        x_data = nu_HDoG_data['local_x']
        y_data = nu_HDoG_data['local_y']
        z_data = nu_HDoG_data['local_z']

        # 途中から撮影しているstackはzの座標を補正する！！！
        if FWorRV == "FW":
            StackPath = os.path.join(self.src_base_dir_FW, self.get_stack_name(
                xx, yy, FWorRV).split("_")[0] + "/" + self.get_stack_name(xx, yy, FWorRV))
        if FWorRV == "RV":
            StackPath = os.path.join(self.src_base_dir_RV, self.get_stack_name(
                xx, yy, FWorRV).split("_")[0] + "/" + self.get_stack_name(xx, yy, FWorRV))
        first_z = int(sorted(os.listdir(StackPath))[0].split(".")[0])
        z_data = z_data + first_z

        if FWorRV == "FW":
            if self.rotCW_FW == 1:  # 反時計回り
                x_data = nu_HDoG_data['local_y']
                y_data = self.x_pixel - nu_HDoG_data['local_x']
            if self.rotCCW_FW == 1:  # 時計回り
                x_data = self.y_pixel - nu_HDoG_data['local_y']
                y_data = nu_HDoG_data['local_x']
            if self.FW_stack_flip_x == 1:
                x_data = self.x_pixel - x_data
            if self.FW_stack_flip_y == 1:
                y_data = self.y_pixel - y_data
        if FWorRV == "RV":
            if self.rotCW_RV == 1:
                x_data = nu_HDoG_data['local_y']
                y_data = self.x_pixel - nu_HDoG_data['local_x']
            if self.rotCCW_RV == 1:
                x_data = self.y_pixel - nu_HDoG_data['local_y']
                y_data = nu_HDoG_data['local_x']
            if self.RV_stack_flip_x == 1:
                x_data = self.x_pixel - x_data
            if self.RV_stack_flip_y == 1:
                y_data = self.y_pixel - y_data
        return np.array([x_data, y_data, z_data])
    
    def load_point_data_all (self, xx, yy, FWorRV): # classifyでpositiveになるデータ以外も含めて全部読み出す。
        #nu_classify_data = self.load_nu_classify_data(xx, yy, FWorRV)
        nu_HDoG_data = self.load_nu_HDoG_data(xx, yy, FWorRV)#[nu_classify_data]
        #print(nu_HDoG_data)
        # Convert to ndarray if it's a list
        if isinstance(nu_HDoG_data, list) or len(nu_HDoG_data) == 0:
            return np.array([], dtype=dt_local)
        else:
            x_data = nu_HDoG_data['local_x']
            y_data = nu_HDoG_data['local_y']
            z_data = nu_HDoG_data['local_z']

            # 途中から撮影しているstackはzの座標を補正する！！！
            if FWorRV == "FW":
                StackPath = os.path.join(self.src_base_dir_FW, self.get_stack_name(
                    xx, yy, FWorRV).split("_")[0] + "/" + self.get_stack_name(xx, yy, FWorRV))
            if FWorRV == "RV":
                StackPath = os.path.join(self.src_base_dir_RV, self.get_stack_name(
                    xx, yy, FWorRV).split("_")[0] + "/" + self.get_stack_name(xx, yy, FWorRV))
            first_z = int(sorted(os.listdir(StackPath))[0].split(".")[0])
            z_data = z_data + first_z

            if FWorRV == "FW":
                if self.rotCW_FW == 1:  # 反時計回り
                    x_data = nu_HDoG_data['local_y']
                    y_data = self.x_pixel - nu_HDoG_data['local_x']
                if self.rotCCW_FW == 1:  # 時計回り
                    x_data = self.y_pixel - nu_HDoG_data['local_y']
                    y_data = nu_HDoG_data['local_x']
                if self.FW_stack_flip_x == 1:
                    x_data = self.x_pixel - x_data
                if self.FW_stack_flip_y == 1:
                    y_data = self.y_pixel - y_data
            if FWorRV == "RV":
                if self.rotCW_RV == 1:
                    x_data = nu_HDoG_data['local_y']
                    y_data = self.x_pixel - nu_HDoG_data['local_x']
                if self.rotCCW_RV == 1:
                    x_data = self.y_pixel - nu_HDoG_data['local_y']
                    y_data = nu_HDoG_data['local_x']
                if self.RV_stack_flip_x == 1:
                    x_data = self.x_pixel - x_data
                if self.RV_stack_flip_y == 1:
                    y_data = self.y_pixel - y_data
            return np.array([x_data, y_data, z_data])

    def RV_LeftUp(self, xxx, yyy):  # 境界の左上のglobal座標
        if xxx == 0:
            if yyy == 0:
                return np.array([self.x_overlap, self.y_overlap])
            if yyy != 0 and yyy != self.y_list_num:
                x_coordinate = max(
                    self.shift_RV[xxx, yyy][0], self.shift_RV[xxx, yyy-1][0]) + self.x_overlap
                y_coordinate = (
                    (self.shift_RV[xxx, yyy-1][1] + self.y_pixel) + self.shift_RV[xxx, yyy][1])/2
                return np.array([x_coordinate, y_coordinate])
        if yyy == 0 and xxx != self.x_list_num:
            x_coordinate = (
                (self.shift_RV[xxx-1, yyy][0] + self.x_pixel) + self.shift_RV[xxx, yyy][0])/2
            y_coordinate = max(
                self.shift_RV[xxx, yyy][1], self.shift_RV[xxx-1, yyy][1]) + self.y_overlap
            return np.array([x_coordinate, self.x_overlap])

        if xxx == self.x_list_num:
            if yyy == self.y_list_num:
                return self.shift_RV[self.x_list_num-1, self.y_list_num-1][:2] + np.array([self.x_pixel, self.y_pixel]) - np.array([self.x_overlap, self.y_overlap])
            if yyy == 0:
                x_coordinate = self.shift_RV[xxx-1,
                                             yyy][0] + self.x_pixel - self.x_overlap
                y_coordinate = self.shift_RV[xxx-1, yyy][1] + self.y_overlap
                return np.array([x_coordinate, y_coordinate])
            if yyy != self.y_list_num and yyy != 0:
                x_coordinate = min(
                    self.shift_RV[xxx-1, yyy][0], self.shift_RV[xxx-1, yyy-1][0]) + self.x_pixel-self.x_overlap
                y_coordinate = (
                    (self.shift_RV[xxx-1, yyy-1][1] + self.y_pixel) + self.shift_RV[xxx-1, yyy][1])/2
                return np.array([x_coordinate, y_coordinate])
        if yyy == self.y_list_num:
            if xxx == 0:
                x_coordinate = self.shift_RV[xxx, yyy-1][0] + self.x_overlap
                y_coordinate = self.shift_RV[xxx, yyy -
                                             1][1] + self.y_pixel - self.y_overlap
                return np.array([x_coordinate, y_coordinate])
            if xxx != 0:
                x_coordinate = (
                    (self.shift_RV[xxx-1, yyy-1][0] + self.x_pixel) + self.shift_RV[xxx, yyy-1][0])/2
                y_coordinate = min(
                    self.shift_RV[xxx-1, yyy-1][1], self.shift_RV[xxx, yyy-1][1]) + self.y_pixel - self.y_overlap
                return np.array([x_coordinate, y_coordinate])

        x_coordinate = (max(self.shift_RV[xxx, yyy][0], self.shift_RV[xxx, yyy-1][0]) + min(
            self.shift_RV[xxx-1, yyy][0]+self.x_pixel, self.shift_RV[xxx-1, yyy-1][0]+self.x_pixel))/2
        y_coordinate = (max(self.shift_RV[xxx, yyy][1], self.shift_RV[xxx-1, yyy][1]) + min(
            self.shift_RV[xxx, yyy-1][1]+self.x_pixel, self.shift_RV[xxx-1, yyy-1][1]+self.y_pixel))/2
        return np.array([x_coordinate, y_coordinate])

    def RV_point_global_with_clipped(self, xx, yy):
        if len(self.load_point_data(xx, yy, "RV")) == 0:
            return np.array([])
        point_data = self.load_point_data(xx, yy, "RV").T + np.array(
            [self.shift_RV[xx, yy][0], self.shift_RV[xx, yy][1], -self.shift_RV[xx, yy][2]])
        point_data = point_data.T
        a, b = line_formula_y(self.RV_LeftUp(xx, yy), self.RV_LeftUp(xx+1, yy))
        point_clf0 = (point_data[1] > a*point_data[0] + b)
        a, b = line_formula_y(self.RV_LeftUp(xx, yy+1),
                              self.RV_LeftUp(xx+1, yy+1))
        point_clf1 = (point_data[1] < a*point_data[0] + b)
        a, b = line_formula_x(self.RV_LeftUp(xx, yy), self.RV_LeftUp(xx, yy+1))
        point_clf2 = (point_data[0] > a*point_data[1] + b)
        a, b = line_formula_x(self.RV_LeftUp(xx+1, yy),
                              self.RV_LeftUp(xx+1, yy+1))
        point_clf3 = (point_data[0] < a*point_data[1] + b)
        # z<0を消す
        point_clf4 = point_data[2] > 0
        point_clf = point_clf0*point_clf1*point_clf2*point_clf3 * point_clf4
        return point_data.T[point_clf].T
    def RV_point_global_with_clipped_all (self, xx, yy): # new
        if len(self.load_point_data_all(xx, yy, "RV")) == 0:
            return np.array([])
        point_data = self.load_point_data_all(xx, yy, "RV").T + np.array(
            [self.shift_RV[xx, yy][0], self.shift_RV[xx, yy][1], -self.shift_RV[xx, yy][2]])
        point_data = point_data.T
        a, b = line_formula_y(self.RV_LeftUp(xx, yy), self.RV_LeftUp(xx+1, yy))
        point_clf0 = (point_data[1] > a*point_data[0] + b)
        a, b = line_formula_y(self.RV_LeftUp(xx, yy+1),
                              self.RV_LeftUp(xx+1, yy+1))
        point_clf1 = (point_data[1] < a*point_data[0] + b)
        a, b = line_formula_x(self.RV_LeftUp(xx, yy), self.RV_LeftUp(xx, yy+1))
        point_clf2 = (point_data[0] > a*point_data[1] + b)
        a, b = line_formula_x(self.RV_LeftUp(xx+1, yy),
                              self.RV_LeftUp(xx+1, yy+1))
        point_clf3 = (point_data[0] < a*point_data[1] + b)
        # z<0を消す
        point_clf4 = point_data[2] > 0
        point_clf = point_clf0*point_clf1*point_clf2*point_clf3 * point_clf4

        # new
        point_data.T[point_clf < 1] = [0,0,0]
        return point_data
    # 回転行列からオイラー角を求める関数
    def rotation_angles_ZYX(self, mat):
        mat = np.array(mat).reshape(3, 3)
        rot = Rotation.from_matrix(mat)
        return rot.as_euler('ZYX') * 180 / np.pi

    # NumpyからAntsImageに変換する関数（X,Zの入れ替え）
    def numpy2ants(self, img):
        img = img.astype(np.float32)
        img = img.transpose(2, 1, 0)
        # components=Falseにすることで3次元の画像として読まれる
        return ants.from_numpy(img, has_components=False)

    # AntsImageからNumpyに変換する関数（X,Zの入れ替え
    def ants2numpy(self, img_ants):
        img_n = img_ants.numpy()
        img_n = img_n.transpose(2, 1, 0)
        return img_n

    # pointのtransformの関数
    def point_transform(self, X_list, Y_list, Z_list, transform):
        center = np.array(transform.GetCenter())
        rotation = np.array(transform.GetMatrix()).reshape(3, 3)
        translation = np.array(transform.GetTranslation())
        ori = np.array([X_list, Y_list, Z_list]).T
        tmp = ori - translation
        tmp = (tmp - center).T
        rotation_inv = np.linalg.inv(rotation)
        tmp = np.dot(rotation_inv, tmp)
        dst = tmp.T + center
        moved_x = np.array(dst.T[0])
        moved_y = np.array(dst.T[1])
        moved_z = np.array(dst.T[2])
        return np.array([moved_x, moved_y, moved_z])


class stitching_FW_RV(stitching_basic_functions):
    def __init__(self, param, stitching_img_um):
        super().__init__(param, stitching_img_um)

    def get_RV_coodinate_before_rotation(self):
        # RVの回転前のそれぞれのstackの中心座標を取得(5umスケール)
        # まずはoriginalのピクセルサイズで
        self.center_RV_ori = self.shift_RV + \
            np.array([int(self.x_pixel/2), int(self.y_pixel/2), 0])
        # 5umスケールのピクセルサイズで
        self.center_RV_5um = self.center_RV_ori*np.array([self.scale_x/self.FW_RV_stitching_scale,
                                                          self.scale_y/self.FW_RV_stitching_scale,
                                                          self.scale_z/self.FW_RV_stitching_scale])
        self.center_RV_5um_int = self.center_RV_5um.astype(np.int32)

        # まずはoriginalのピクセルサイズで
        self.shift_RV_ori = self.shift_RV
        # 5umスケールのピクセルサイズで
        self.shift_RV_5um = self.shift_RV_ori*np.array([self.scale_x/self.FW_RV_stitching_scale,
                                                        self.scale_y/self.FW_RV_stitching_scale,
                                                        self.scale_z/self.FW_RV_stitching_scale])
        self.shift_RV_5um_int = self.shift_RV_5um.astype(np.int32)
        # self.stack_5um = int(self.x_pixel*self.scale_x/self.FW_RV_stitching_scale)#5umスケールでのstackの一辺のpixel
        return

    # ANTs用の画像をresizeして保存する関数（FWは5枚、RVは枚）
    def save_image_for_rotation(self, Rnum):
        FW_RV_1st_X, FW_RV_1st_Y = self.rasen_xy[Rnum]

        micro = self.FW_RV_stitching_scale  # must int*self.z_scale(2.5)
        z_step = int(micro/self.scale_z)  # 2

        # FW_loadimage
        Maisuu = self.FW_ants_maisuu  # must odd number --5
        z_start = int(self.FW_boundary - ((Maisuu-1)/2)*z_step)
        z_end = int(self.FW_boundary + ((Maisuu-1)/2)*z_step)+1
        FW_imgs = np.array([self.import_img_xyz(FW_RV_1st_X, FW_RV_1st_Y, iii, "FW")
                           for iii in range(z_start, z_end, z_step)])[::-1]

        # RV_loadimage
        Maisuu = self.RV_ants_maisuu  # must odd number --151
        z_start = int(self.RV_boundary - ((Maisuu-1)/2)*z_step)
        z_end = int(self.RV_boundary + ((Maisuu-1)/2)*z_step)+1
        RV_imgs = np.array([self.import_img_xyz(
            FW_RV_1st_X, FW_RV_1st_Y, iii, "RV") for iii in range(z_start, z_end, z_step)])

        # 5um scale
        FW_resize = np.array([cv2.resize(Fimg, dsize=None, fx=self.scale_x /
                             micro, fy=self.scale_y/micro) for Fimg in FW_imgs]).astype('float32')
        RV_resize = np.array([cv2.resize(Rimg, dsize=None, fx=self.scale_x /
                             micro, fy=self.scale_y/micro) for Rimg in RV_imgs]).astype('float32')

        save_FWimg_path = os.path.join(self.stitching_check_path, "FW_target/X{0}_Y{1}.tif".format(
            str(FW_RV_1st_X).zfill(4), str(FW_RV_1st_Y).zfill(4)))
        tifffile.imsave(save_FWimg_path, FW_resize.astype(np.uint16))
        save_RVimg_path = os.path.join(self.stitching_check_path, "RV_moving/X{0}_Y{1}.tif".format(
            str(FW_RV_1st_X).zfill(4), str(FW_RV_1st_Y).zfill(4)))
        tifffile.imsave(save_RVimg_path, RV_resize.astype(np.uint16))
        return

    # 回転行列を取得するための関数
    def rotation_matrix(self, FW_RV_X, FW_RV_Y):
        micro = self.FW_RV_stitching_scale  # 5
        ut1 = time.time()
        print("FW_RV_X = ", FW_RV_X, "FW_RV_Y = ", FW_RV_Y)
        # reding images
        FW_RV_1st_X = FW_RV_X
        FW_RV_1st_Y = FW_RV_Y

        save_img_path = os.path.join(self.stitching_check_path, "FW_target/X{0}_Y{1}.tif".format(
            str(FW_RV_1st_X).zfill(4), str(FW_RV_1st_Y).zfill(4)))
        FW_resize = tifffile.imread(save_img_path)
        FW_resize = FW_resize.astype(np.float32)

        save_img_path = os.path.join(self.stitching_check_path, "RV_moving/X{0}_Y{1}.tif".format(
            str(FW_RV_1st_X).zfill(4), str(FW_RV_1st_Y).zfill(4)))
        RV_resize = tifffile.imread(save_img_path)
        RV_resize = RV_resize.astype(np.float32)

        moving_img = self.numpy2ants(RV_resize)
        target_img = self.numpy2ants(FW_resize)
        out_ants = ants.registration(
            fixed=target_img, moving=moving_img, type_of_transform='Rigid')
        warped = out_ants['warpedmovout']

        save_img_path = os.path.join(self.stitching_check_path, "RV_moved/X{0}_Y{1}.tif".format(
            str(FW_RV_1st_X).zfill(4), str(FW_RV_1st_Y).zfill(4)))
        tifffile.imsave(save_img_path, self.ants2numpy(
            warped).astype(np.uint16))
        if len(out_ants['fwdtransforms']) == 0:
            dammy_rotation_center = (10000, 10000, 10000)
            dammy_axis = (0, 0, 1)
            dammy_angle = np.pi
            dammy_translation = (1, 2, 3)
            dammy_scale_factor = 2.0
            dammy_Transform = sitk.Similarity3DTransform(
                dammy_scale_factor, dammy_axis, dammy_angle, dammy_translation, dammy_rotation_center)
            ut2 = time.time()
            print("dammy_ANTs", ut2-ut1)
            return dammy_Transform

        Transform = sitk.ReadTransform(out_ants['fwdtransforms'][0])
        Transform = sitk.AffineTransform(Transform)
        ut2 = time.time()
        print("ANTs", ut2-ut1)
        return Transform

    # 回転行列を計算して保存（RVの枚数は傾きにより計算→枚数によってtransformのcenterのzが変わることに注意！）
    def calc_large_rotation_transform(self):
        ut1 = time.time()
        rasen_check_range = self.rasen_check_range  # 25
        # rotaion_matrixを決めるための画像を保存（rasen1-25）
        making_images_process = joblib.Parallel(
            n_jobs=-1)(joblib.delayed(self.save_image_for_rotation)(ii) for ii in range(rasen_check_range))

        # ANTs_1st
        transform_list_1st = [self.rotation_matrix(
            self.rasen_xy[ii][0], self.rasen_xy[ii][1]) for ii in range(rasen_check_range)]

        # rotationのオイラー角のリストを作成（単位は°に変換）[rotation_list_1st]
        rotation_list_1st = np.array([self.rotation_angles_ZYX(
            transform_list_1st[ii].GetMatrix()) for ii in range(rasen_check_range)])

        # ANTs_2nd(安定して同じ値を取るstackを探す)
        transform_list_2nd = [self.rotation_matrix(
            self.rasen_xy[ii][0], self.rasen_xy[ii][1]) for ii in range(rasen_check_range)]

        # rotationのオイラー角のリストを作成（単位は°に変換）[rotation_list_2nd]
        rotation_list_2nd = np.array([self.rotation_angles_ZYX(
            transform_list_1st[ii].GetMatrix()) for ii in range(rasen_check_range)])

        not_dammy_list = np.array(rotation_list_1st.T[0] != 180)
        print("not_dammy_list\n", not_dammy_list)

        # variabilityは二回の計算の誤差のx,y,zの合計が0.1以下のindex
        variability = np.sum(
            np.abs(rotation_list_2nd-rotation_list_1st), axis=1) < 0.1
        print("same_calc_stack\n", variability)
        variability = variability * not_dammy_list
        print("same_calc_stack in not_dammy_list\n", variability)
        # うまく合うスタックがないとassert
        assert np.sum(variability) != 0, "There is no FW_RV_ANTs matching stacks in range {}".format(
            rasen_check_range)
        saishou_candidate = np.arange(rasen_check_range)[variability]
        rotation_list_candidate = rotation_list_1st[variability]

        # rotationの角度の中央値を取得
        euler = np.median(rotation_list_candidate, axis=0)
        print(rotation_list_candidate, euler)
        # 中央値からの変化がすくないスタックを選択（角度0.01から0.2まで調べる）
        center_stack_rasen_num = saishou_candidate[np.argmin(
            np.sum(np.abs(rotation_list_candidate-euler), axis=1))]
        rotation_center_stack_x, rotation_center_stack_y = self.rasen_xy[center_stack_rasen_num]
        self.rotation_center_stack_x, self.rotation_center_stack_y, self.rotation_center_stack_rasen_num = rotation_center_stack_x, rotation_center_stack_y, center_stack_rasen_num
        print("rotation center = ", "X = ", self.rotation_center_stack_x,
              ", Y = ", self.rotation_center_stack_y)
        Transform_center_5um = transform_list_1st[center_stack_rasen_num]
        ut2 = time.time()
        print("{}stacks of rough ants takes ".format(
            str(rasen_check_range)), ut2-ut1)

        # FW_boundaryに本当にあっているzの枚数(FWを5枚読み込んだので+2,5umspanなので*2)
        z_step = int(self.FW_RV_stitching_scale/self.scale_z)
        center_z_of_FWimgs = int((self.FW_ants_maisuu - 1)/2)
        center_z_of_RVimgs = int((self.RV_ants_maisuu - 1)/2)
        self.RV_boundary_hosei = self.RV_boundary - z_step*center_z_of_RVimgs + \
            z_step*(int(Transform_center_5um.GetTranslation()
                    [2])+center_z_of_FWimgs)

        # RV_point_transferのために保存
        print("save rotation center and RV_boundary_hosei（FW_RV_stitchingでずれた分補正している）")
        print("RV_bundary = ", self.RV_boundary, ",",
              "RV_boundary_hosei = ", self.RV_boundary_hosei)
        np.save(os.path.join(self.stitching_check_path, "rotation_center_stack_and_RV_boundary_hosei.npy"), np.array(
            [rotation_center_stack_x, rotation_center_stack_y, self.RV_boundary_hosei]))

        # 0.65umでのANTs
        x_num, y_num, z_num_FW, z_num_RV = self.rotation_center_stack_x, self.rotation_center_stack_y, self.FW_boundary, self.RV_boundary_hosei
        ut1 = time.time()
        self.FW_ants_maisuu065 = 5
        FW_img_np = np.array([self.import_img_xyz(x_num, y_num, z_num_FW+(ii-int(
            self.FW_ants_maisuu065/2)), "FW") for ii in range(self.FW_ants_maisuu065)])[::-1]
        self.RV_ants_maisuu065 = 151
        self.RV_ants_maisuu065_half = int(self.RV_ants_maisuu065/2)
        np.save(os.path.join(self.stitching_check_path, "RV_ants_maisuu065.npy"), np.array(
            [self.RV_ants_maisuu065, self.RV_ants_maisuu065_half]))

        RV_img_np = np.array([self.import_img_xyz(x_num, y_num, z_num_RV+(ii-int(
            self.RV_ants_maisuu065/2)), "RV") for ii in range(self.RV_ants_maisuu065)])
        niiFW_img = tif2nii_no_save(
            FW_img_np, self.scale_x, self.scale_y, self.scale_z)
        niiRV_img = tif2nii_no_save(
            RV_img_np, self.scale_x, self.scale_y, self.scale_z)
        antsFW_img = ants.utils.nifti_to_ants(niiFW_img)
        save_img_path = os.path.join(
            self.stitching_check_path, "FW_target_065.tif")
        tifffile.imsave(save_img_path, self.ants2numpy(
            antsFW_img).astype(np.uint16))
        antsRV_img = ants.utils.nifti_to_ants(niiRV_img)
        save_img_path = os.path.join(
            self.stitching_check_path, "RV_moving_065.tif")
        tifffile.imsave(save_img_path, self.ants2numpy(
            antsRV_img).astype(np.uint16))
        del FW_img_np, RV_img_np, niiFW_img, niiRV_img
        ut2 = time.time()
        print("0.65 ants iamge preparing takes ", ut2-ut1)
        ut1 = time.time()
        print("0.65um ants registration...")
        out_ants_065 = ants.registration(
            fixed=antsFW_img, moving=antsRV_img, type_of_transform='Rigid')
        ut2 = time.time()
        print("0.65um ants takes ", ut2-ut1)
        warped = out_ants_065['warpedmovout']
        save_img_path = os.path.join(
            self.stitching_check_path, "RV_moved_065.tif")
        tifffile.imsave(save_img_path, self.ants2numpy(
            warped).astype(np.uint16))

        Transform_065 = sitk.ReadTransform(out_ants_065['fwdtransforms'][0])
        Transform_065 = sitk.AffineTransform(Transform_065)
        Transform_large = sitk.AffineTransform(Transform_065)
        print("rotation_angle_of_0.65um_scale_ANTs",
              self.rotation_angles_ZYX(Transform_large.GetMatrix()))
        delta_Z = 0
        delta_Y = self.shift_RV[x_num, y_num][1]*self.scale_y
        delta_X = self.shift_RV[x_num, y_num][0]*self.scale_x
        new_center = Transform_large.GetCenter(
        )+np.array([delta_X, delta_Y, delta_Z])
        Transform_large.SetCenter(new_center)
        self.Transform_large = Transform_large
        transform_large_path = os.path.join(
            self.stitching_dst_path, "Transform_large_original.mat")
        sitk.WriteTransform(Transform_large, transform_large_path)

    # BeforeStartZum, AfterStartZumの原点はRotation centerの1枚目
    def Transform_large_Zum(self, BeforeStartZum, AfterStartZum):
        transform_large_path = os.path.join(
            self.stitching_dst_path, "Transform_large_original.mat")
        Transform_large = sitk.AffineTransform(
            sitk.ReadTransform(transform_large_path))
        Transform_large_Zum = sitk.AffineTransform(Transform_large)
        Tori, Cori = Transform_large.GetTranslation(
        )[2], Transform_large.GetCenter()[2]
        C = self.RV_boundary_hosei*self.scale_z - self.RV_ants_maisuu065_half * \
            self.scale_z + Tori+Cori - AfterStartZum
        T = AfterStartZum - BeforeStartZum
        NewCenter = Transform_large.GetCenter() + np.array([0, 0, C-Cori])
        NewTranslation = Transform_large.GetTranslation() + \
            np.array([0, 0, T-Tori])
        Transform_large_Zum.SetCenter(NewCenter)
        Transform_large_Zum.SetTranslation(NewTranslation)
        return Transform_large_Zum

    def rotation_of_all_RV_points(self):
        ut1 = time.time()
        RV_all_points = np.empty((3, 0))
        for xx in range(self.x_list_num):
            for yy in range(self.y_list_num):
                if self.RV_point_global_with_clipped(xx, yy).size != 0:
                    RV_all_points = np.concatenate(
                        [RV_all_points, self.RV_point_global_with_clipped(xx, yy)], 1)
        ut2 = time.time()
        print("RV_points load takes ", ut2-ut1)

        x_data = RV_all_points[0]
        y_data = RV_all_points[1]
        # pointのzの原点をrotation centerにずらしている！！！
        if os.path.exists(os.path.join(self.stitching_check_path, "rotation_center_stack_x_and_y.npy")):
            self.rotation_center_stack_x, self.rotation_center_stack_y, self.RV_boundary_hosei = np.load(
                os.path.join(self.stitching_check_path, "rotation_center_stack_and_RV_boundary_hosei.npy"))
        z_data = RV_all_points[2] + self.shift_RV[self.rotation_center_stack_x,
                                                  self.rotation_center_stack_y][2]

        x_data_all_um = (x_data)*self.scale_x
        y_data_all_um = (y_data)*self.scale_y
        z_data_all_um = (z_data)*self.scale_z

        # ｚの原点はboundary
        R = self.point_transform(x_data_all_um, y_data_all_um, z_data_all_um,
                                 self.Transform_large_Zum(0, self.RV_boundary_hosei*self.scale_z))
        self.roteted_RV_points = R
    def save_stitched_RV_stack_data_all (self, xx, yy, boundary_um): # new
        RV_all_points_um = np.empty((3, 0))
        if self.RV_point_global_with_clipped_all(xx, yy).size == 0:
            print("emp_r")
        else:
            tmp=self.RV_point_global_with_clipped_all(xx, yy)
            x_data = tmp[0]
            y_data = tmp[1]
            
            # pointのzの原点をrotation centerにずらしている！！！
            if os.path.exists(os.path.join(self.stitching_check_path, "rotation_center_stack_x_and_y.npy")):
                self.rotation_center_stack_x, self.rotation_center_stack_y, self.RV_boundary_hosei = np.load(
                    os.path.join(self.stitching_check_path, "rotation_center_stack_and_RV_boundary_hosei.npy"))
            z_data = tmp[2] + self.shift_RV[self.rotation_center_stack_x,
                                                    self.rotation_center_stack_y][2]

            x_data_all_um = (x_data)*self.scale_x
            y_data_all_um = (y_data)*self.scale_y
            z_data_all_um = (z_data)*self.scale_z

            # ｚの原点はboundary
            R2 = self.point_transform(x_data_all_um, y_data_all_um, z_data_all_um,
                                    self.Transform_large_Zum(0, self.RV_boundary_hosei*self.scale_z))

            #RV_all_points_um_tmp = R2.T[R2[2] > 0].T
            
            R2.T[R2[2] <= 0] = [0,0,(-1)*boundary_um] # lenは変えない ここである

            RV_all_points_um = np.array(
                [R2[0], R2[1], boundary_um + R2[2]])

        self.save_stitched_stack_data(xx, yy, "RV", RV_all_points_um)
            
            #self.roteted_RV_points = R
    # RV_maisuu
    def calc_RV_maisuu(self):
        center_Xs = self.center_RV_ori[:, :, 0].flatten()*self.scale_x
        center_Ys = self.center_RV_ori[:, :, 1].flatten()*self.scale_y
        center_Zs = np.ones(len(center_Xs))*self.RV_boundary_hosei*self.scale_z

        Rotated_centers = self.point_transform(
            center_Xs, center_Ys, center_Zs, self.Transform_large_Zum(0, self.RV_boundary_hosei*self.scale_z))
        self.RV_maisuu_half_5um = np.ceil(max(abs(np.min(Rotated_centers[2])), abs(
            np.max(Rotated_centers[2])))/self.FW_RV_stitching_scale).astype(np.uint16)
        np.save(os.path.join(self.stitching_check_path,
                "RV_maisuu_half_5um.npy"), np.array([self.RV_maisuu_half_5um]))

        center_X = self.center_RV_ori[self.rotation_center_stack_x,
                                      self.rotation_center_stack_y, 0]*self.scale_x
        center_Y = self.center_RV_ori[self.rotation_center_stack_x,
                                      self.rotation_center_stack_y, 1]*self.scale_y
        center_Z = np.ones(1)*self.RV_boundary_hosei*self.scale_z
        self.Rotated_center_um = self.point_transform(
            center_X, center_Y, center_Z, self.Transform_large_Zum(0, self.RV_boundary_hosei*self.scale_z))

    # rotation centerのz座標でのRVのbig imageを作る関数
    def make_big_image_RV(self, ZZZ, resize_um):
        shift_RV_risize = self.shift_RV * \
            np.array([self.scale_x/resize_um, self.scale_y/resize_um, 1])
        shift_RV_risize_int = shift_RV_risize.astype(np.int64)
        um = resize_um
        big_img = np.zeros((int(self.y_pixel*(self.y_list_num+1)/(resize_um / self.scale_x)),
                           int(self.x_pixel*(self.x_list_num+1)/(resize_um / self.scale_x))))
        for yy in range(self.y_list_num):
            for xx in range(self.x_list_num):
                # big imageのzの原点をrotation centerに！！！
                img = (self.import_img_xyz(
                    xx, yy, ZZZ+self.shift_RV[xx, yy][2] - self.shift_RV[self.rotation_center_stack_x, self.rotation_center_stack_y][2], "RV")).astype("float32")
                img_patch = cv2.resize(
                    img, dsize=None, fx=self.scale_x/um, fy=self.scale_y/um, interpolation=cv2.INTER_AREA)
                img_patch = img_patch[self.down_norishiro:, :]
                img_size = img_patch.shape
                big_img[shift_RV_risize_int[xx, yy, 1]+self.down_norishiro:shift_RV_risize_int[xx, yy, 1]+self.down_norishiro +
                        img_size[0], shift_RV_risize_int[xx, yy, 0]:shift_RV_risize_int[xx, yy, 0]+img_size[1]] = img_patch
        return big_img

    # rotation centerのz座標でのRVのbig imageを作る関数
    def make_big_image_RV_norm(self, ZZZ, resize_um):
        shift_RV_risize = self.shift_RV * \
            np.array([self.scale_x/resize_um, self.scale_y/resize_um, 1])
        shift_RV_risize_int = shift_RV_risize.astype(np.int64)
        um = resize_um
        big_img = np.zeros((int(self.y_pixel*(self.y_list_num+1)/(resize_um / self.scale_x)),
                           int(self.x_pixel*(self.x_list_num+1)/(resize_um / self.scale_x))))
        for yy in range(self.y_list_num):
            for xx in range(self.x_list_num):
                # big imageのzの原点をrotation centerに！！！
                img = (self.norm_large(
                    xx, yy, ZZZ+self.shift_RV[xx, yy][2] - self.shift_RV[self.rotation_center_stack_x, self.rotation_center_stack_y][2], "RV")).astype("float32")
                img_patch = cv2.resize(
                    img, dsize=None, fx=self.scale_x/um, fy=self.scale_y/um, interpolation=cv2.INTER_AREA)
                img_patch = img_patch[self.down_norishiro:, :]
                img_size = img_patch.shape
                big_img[shift_RV_risize_int[xx, yy, 1]+self.down_norishiro:shift_RV_risize_int[xx, yy, 1]+img_size[0] +
                        self.down_norishiro, shift_RV_risize_int[xx, yy, 0]:shift_RV_risize_int[xx, yy, 0]+img_size[1]] = img_patch
        del img, img_patch
        gc.collect()
        return big_img

    def getRVtarget(self, XX, YY, large_image_boundary_065):
        CX_0, CY_0 = int(
            self.Rotated_center_um[0]/0.65), int(self.Rotated_center_um[1]/0.65)
        dX, dY = XX-self.rotation_center_stack_x, YY-self.rotation_center_stack_y
        CX = CX_0+(self.x_pixel-self.x_overlap*2)*dX
        CY = CY_0+(self.y_pixel-self.y_overlap*2)*dY
        img_gry = large_image_boundary_065[int(
            CY)-1500:int(CY)+1500, int(CX)-1500:int(CX)+1500]
        return img_gry, int(CX)-1500, int(CY)-1500

    def stitchable_FW_RV(self, XXX, YYY, large_image_boundary_065):
        RV_img = self.getRVtarget(XXX, YYY, large_image_boundary_065)[0]
        RV_core = RV_img[600:2400, 600:2400]
        # return np.sum(RV_core>ST_FW_RV.bcg_intensity)>1000
        return np.sum(RV_core > 0) > 1800*900  # 画像の半分はnormの閾値以上

    # FW_RV stitchingの結果をそのまま読み込み[stitchin_clas.pyに移動]
    def FW_RV_stitching_data_raw(self, XX, YY):
        stitching_file = os.path.join(os.path.join(
            self.stitching_dst_path, "FW_RV_stitching_result/"), "X"+str(XX).zfill(5)+"_Y"+str(YY).zfill(5)+".npy")
        return np.load(stitching_file)

    # FW_RV stitchingで計算されなかったところを埋める
    def fill_not_calculated_FW_RV_stitching_stack(self, shift_FW):
        done_FW_RV = np.zeros((self.y_list_num, self.x_list_num))
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)
        FW_RV_stitching_data_np = np.zeros(
            (self.x_list_num, self.y_list_num, 3))

        FW_RV_accuracy_cut_off = self.FW_RV_accuracy_cut_off
        print("FW_RV_accuracy_cut_off = ", FW_RV_accuracy_cut_off)

        for xx in range(self.x_list_num):
            for yy in range(self.y_list_num):
                DD = self.FW_RV_stitching_data_raw(xx, yy)
                if DD[3] > FW_RV_accuracy_cut_off:
                    done_FW_RV[yy, xx] = 1
                    FW_RV_stitching_data_np[xx, yy] = DD[:3]
        tifffile.imsave(os.path.join(self.stitching_check_path,
                        "first_done_FW_RV.tif"), done_FW_RV.astype(np.uint16))
        while (np.sum(done_FW_RV) < self.x_list_num*self.y_list_num):
            next_FW_RV = cv2.dilate(
                done_FW_RV, kernel, iterations=1)-done_FW_RV
            next_FW_RV_index = np.array(np.where(next_FW_RV != 0)).T
            for yy, xx in next_FW_RV_index:
                if xx != 0 and done_FW_RV[yy, xx-1] == 1:
                    done_FW_RV[yy, xx] = 1
                    FW_RV_stitching_data_np[xx, yy] = FW_RV_stitching_data_np[xx-1, yy] + (
                        shift_FW[xx, yy] - shift_FW[xx-1, yy])
                else:
                    if yy != 0 and done_FW_RV[yy-1, xx] == 1:
                        done_FW_RV[yy, xx] = 1
                        FW_RV_stitching_data_np[xx, yy] = FW_RV_stitching_data_np[xx, yy-1] + (
                            shift_FW[xx, yy] - shift_FW[xx, yy-1])
                    else:
                        if xx != self.x_list_num-1 and done_FW_RV[yy, xx+1] == 1:
                            done_FW_RV[yy, xx] = 1
                            FW_RV_stitching_data_np[xx, yy] = FW_RV_stitching_data_np[xx+1, yy] + (
                                shift_FW[xx, yy] - shift_FW[xx+1, yy])
                        else:
                            if yy != self.y_list_num-1 and done_FW_RV[yy+1, xx] == 1:
                                done_FW_RV[yy, xx] = 1
                                FW_RV_stitching_data_np[xx, yy] = FW_RV_stitching_data_np[xx, yy+1] + (
                                    shift_FW[xx, yy] - shift_FW[xx, yy+1])
        self.FW_RV_stitching_data_np = FW_RV_stitching_data_np
        return

        # FW_RV stitchingで計算されなかったところを埋める
    def fill_not_calculated_FW_RV_stitching_stack_nozoku(self, shift_FW, FW_RV_nozoku_list):
        done_FW_RV = np.zeros((self.y_list_num, self.x_list_num))
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)
        FW_RV_stitching_data_np = np.zeros(
            (self.x_list_num, self.y_list_num, 3))

        FW_RV_accuracy_cut_off = self.FW_RV_accuracy_cut_off
        print("FW_RV_accuracy_cut_off = ", FW_RV_accuracy_cut_off)

        for xx in range(self.x_list_num):
            for yy in range(self.y_list_num):
                DD = self.FW_RV_stitching_data_raw(xx, yy)
                if DD[3] > FW_RV_accuracy_cut_off:
                    done_FW_RV[yy, xx] = 1
                    FW_RV_stitching_data_np[xx, yy] = DD[:3]
        for nozoku_stack in FW_RV_nozoku_list:
            done_FW_RV[nozoku_stack[1], nozoku_stack[0]] = 0

        tifffile.imsave(os.path.join(self.stitching_check_path,
                        "first_done_FW_RV_nozoku.tif"), done_FW_RV.astype(np.uint16))
        while (np.sum(done_FW_RV) < self.x_list_num*self.y_list_num):
            next_FW_RV = cv2.dilate(
                done_FW_RV, kernel, iterations=1)-done_FW_RV
            next_FW_RV_index = np.array(np.where(next_FW_RV != 0)).T
            for yy, xx in next_FW_RV_index:
                if xx != 0 and done_FW_RV[yy, xx-1] == 1:  # look_Left
                    done_FW_RV[yy, xx] = 1
                    FW_RV_stitching_data_np[xx, yy] = FW_RV_stitching_data_np[xx-1, yy] + (
                        shift_FW[xx, yy] - shift_FW[xx-1, yy])
                else:
                    if yy != 0 and done_FW_RV[yy-1, xx] == 1:  # look_Up
                        done_FW_RV[yy, xx] = 1
                        FW_RV_stitching_data_np[xx, yy] = FW_RV_stitching_data_np[xx, yy-1] + (
                            shift_FW[xx, yy] - shift_FW[xx, yy-1])
                    else:
                        # look_Right
                        if xx != self.x_list_num-1 and done_FW_RV[yy, xx+1] == 1:
                            done_FW_RV[yy, xx] = 1
                            FW_RV_stitching_data_np[xx, yy] = FW_RV_stitching_data_np[xx+1, yy] + (
                                shift_FW[xx, yy] - shift_FW[xx+1, yy])
                        else:
                            # look_Down
                            if yy != self.y_list_num-1 and done_FW_RV[yy+1, xx] == 1:
                                done_FW_RV[yy, xx] = 1
                                FW_RV_stitching_data_np[xx, yy] = FW_RV_stitching_data_np[xx, yy+1] + (
                                    shift_FW[xx, yy] - shift_FW[xx, yy+1])
        self.FW_RV_stitching_data_np = FW_RV_stitching_data_np
        return

    def image_under_merge_z(self, x_num, y_num, z_at_center):
        ratio = (z_at_center - self.z_start_at_center) / \
            (self.merge_z_at_center - self.z_start_at_center)
        hidariue_at_merge_z = self.shift_FW[x_num, y_num][:2] + self.FW_RV_stitching_data(
            self.first_fill[0], self.first_fill[1])[:2] - self.shift_FW[self.first_fill[0], self.first_fill[1]][:2]
        hidariue_at_z_start = self.FW_RV_stitching_data(x_num, y_num)[:2]
        hidariue_pixel = (hidariue_at_z_start + (ratio *
                          (hidariue_at_merge_z - hidariue_at_z_start))).astype(np.int32)
        z_at_merge_z = self.merge_z_at_center + \
            self.shift_FW[x_num, y_num][2] - \
            self.shift_FW[self.first_fill[0], self.first_fill[1]][2]
        z_start = self.FW_RV_stitching_data(x_num, y_num)[2]
        yomikomu_z = (z_start + (ratio * (z_at_merge_z - z_start))
                      ).astype(np.int32)
        return hidariue_pixel, yomikomu_z

    # FW_big_imagesの作成(norm_large)
    # rotation centerのz座標(枚数)でのRVのbig imageを作る関数
    def make_big_image_FW_norm(self, ZZZ, resize_um,):
        #print("IN")
        shift_RV_risize = self.shift_RV * \
            np.array([self.scale_x/resize_um, self.scale_y/resize_um, 1])
        shift_RV_risize_int = shift_RV_risize.astype(np.int64)
        um = resize_um
        big_img = np.zeros((int(self.y_pixel*(self.y_list_num+1)/(resize_um / self.scale_x)),
                           int(self.x_pixel*(self.x_list_num+1)/(resize_um / self.scale_x))))
        for yy in range(self.y_list_num):
            for xx in range(self.x_list_num):
                # big imageのzの原点をrotation centerに！！！
                hidariue, yomikomu_z = self.image_under_merge_z(xx, yy, ZZZ)
                #print("Y_X: " + str(yy)+ "_" +str(xx) +" hidariue: " + str(hidariue)+" yomikomu_z: " + str(yomikomu_z))
                hidariue_resize = (hidariue*0.65/resize_um).astype(np.int64)
                img = self.norm_large(xx, yy, yomikomu_z,
                                      "FW").astype("float32")
                img_patch = cv2.resize(
                    img, dsize=None, fx=self.scale_x/um, fy=self.scale_y/um, interpolation=cv2.INTER_AREA)
                img_patch = img_patch[self.down_norishiro:, :]
                img_size = img_patch.shape
                big_img[hidariue_resize[1]+self.down_norishiro:hidariue_resize[1]+self.down_norishiro +
                        img_size[0], hidariue_resize[0]:hidariue_resize[0]+img_size[1]] = img_patch
        return big_img

    # FW_big_imagesの作成(norm_large)
    # rotation centerのz座標(枚数)でのRVのbig imageを作る関数
    def make_big_image_FW_norm_line(self, ZZZ, resize_um):
        shift_RV_risize = self.shift_RV * \
            np.array([self.scale_x/resize_um, self.scale_y/resize_um, 1])
        shift_RV_risize_int = shift_RV_risize.astype(np.int64)
        um = resize_um
        big_img = np.zeros((int(self.y_pixel*(self.y_list_num+1)/(resize_um / self.scale_x)),
                           int(self.x_pixel*(self.x_list_num+1)/(resize_um / self.scale_x))))
        for yy in range(self.y_list_num):
            for xx in range(self.x_list_num):
                # big imageのzの原点をrotation centerに！！！
                hidariue, yomikomu_z = self.image_under_merge_z(xx, yy, ZZZ)
                hidariue_resize = (hidariue*0.65/resize_um).astype(np.int64)
                img = self.norm_large(xx, yy, yomikomu_z,
                                      "FW").astype("float32")
                img_patch = cv2.resize(
                    img, dsize=None, fx=self.scale_x/um, fy=self.scale_y/um, interpolation=cv2.INTER_AREA)
                img_patch[0:self.white_line_width, :] = 1
                img_patch[:, 0:self.white_line_width] = 1
                img_size = img_patch.shape
                big_img[hidariue_resize[1]:hidariue_resize[1]+img_size[0],
                        hidariue_resize[0]:hidariue_resize[0]+img_size[1]] = img_patch
        return big_img

    # FW_big_imagesの作成(norm_large)
    # rotation centerのz座標(枚数)でのRVのbig imageを作る関数
    def make_big_image_FW(self, ZZZ, resize_um,):
        shift_RV_risize = self.shift_RV * \
            np.array([self.scale_x/resize_um, self.scale_y/resize_um, 1])
        shift_RV_risize_int = shift_RV_risize.astype(np.int64)
        um = resize_um
        big_img = np.zeros((int(self.y_pixel*(self.y_list_num+1)/(resize_um / self.scale_x)),
                           int(self.x_pixel*(self.x_list_num+1)/(resize_um / self.scale_x))))
        for yy in range(self.y_list_num):
            for xx in range(self.x_list_num):
                # big imageのzの原点をrotation centerに！！！
                hidariue, yomikomu_z = self.image_under_merge_z(xx, yy, ZZZ)
                hidariue_resize = (hidariue*0.65/resize_um).astype(np.int64)
                img = self.import_img_xyz(
                    xx, yy, yomikomu_z, "FW").astype("float32")
                img_patch = cv2.resize(
                    img, dsize=None, fx=self.scale_x/um, fy=self.scale_y/um, interpolation=cv2.INTER_AREA)
                img_patch = img_patch[self.down_norishiro:, :]
                img_size = img_patch.shape
                big_img[hidariue_resize[1]+self.down_norishiro:hidariue_resize[1]+self.down_norishiro +
                        img_size[0], hidariue_resize[0]:hidariue_resize[0]+img_size[1]] = img_patch
        return big_img
