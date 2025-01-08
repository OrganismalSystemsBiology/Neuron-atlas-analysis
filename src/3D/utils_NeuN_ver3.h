#ifndef HDOG_UTILS_H
#define HDOG_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "stdio.h"
#include <json.hpp>

using json = nlohmann::json;

namespace param {
  struct SigmaDoG {
    float small_xy;
    float small_z;
    float large_xy;
    float large_z;
  };
  struct RadiusNorm {
    int large_xy;
    int large_z;
  };
  void to_json(json& j, const SigmaDoG& p) {
    j = json{{"xy_small", p.small_xy},
             {"z_small", p.small_z},
             {"xy_large", p.large_xy},
             {"z_large", p.large_z}};
  }
  void from_json(const json& j, SigmaDoG& p) {
    j.at("xy_small").get_to(p.small_xy);
    j.at("z_small").get_to(p.small_z);
    j.at("xy_large").get_to(p.large_xy);
    j.at("z_large").get_to(p.large_z);
  }
  void to_json(json& j, const RadiusNorm& p) {
    j = json{{"xy_large", p.large_xy},
             {"z_large", p.large_z}};
  }
  void from_json(const json& j, RadiusNorm& p) {
    j.at("xy_large").get_to(p.large_xy);
    j.at("z_large").get_to(p.large_z);
  }
}

/*  Parameters */
struct Parameters {
  // GPU Device ID
  int devID;
  bool verbose = false;

  // size [pixel]
  int width = 2048;
  int height = 2048;
  int depth = 32;
  int image_size; // = width * height * depth
  int image_size2D; // = width*height
  // cub_tmp_size_factor = 8.001
  // assertion fail if this is smaller than required
  float cub_tmp_size_factor = 8.7;//8.001;
  size_t cub_tmp_bytes; // = image_size * cub_tmp_size_factor

  // margin [pixel]
  int left_margin = 74;
  int right_margin = 74;
  int top_margin = 74;
  int bottom_margin = 74;
  int depth_margin; // automatically determined
  int extra_depth_margin = 3;

  // scale [um/pixel] nagative value to flip the direction
  float scale_x = 1.0;
  float scale_y = 1.0;
  float scale_z = 1.0;

  // algorithm parameters
  float gamma_n = 1.0;
  param::SigmaDoG sigma_dog;
  param::RadiusNorm radius_norm;
  unsigned short min_intensity_skip = 1000;
  unsigned short min_intensity_truncate = 1000;
  unsigned short min_intensity_truncate_Iba1 = 1000;
  unsigned short min_intensity_truncate_Iba1_left = 1000;
  unsigned short min_intensity_truncate_NeuN = 1000;
  bool is_ave_mode = false;

  // threshold to reduce result filesize
  unsigned short min_size = 0;

  //new
  int center = 10000;
  int offset_right_NeuN = 0;
  int offset_right_Iba1 = 0;
  int offset_left_NeuN = 0;
  int offset_left_Iba1 = 0;

  // stacks
  int n_stack;
  std::vector<std::vector<std::string> > list_src_path;
  std::vector<int> list_stack_length;
  std::vector<std::string> list_dst_path;

  // parameter search
  int n_search;
  std::vector<param::RadiusNorm> list_radius_norm;
  std::vector<param::SigmaDoG> list_sigma_dog;
};

void loadParamFile(const std::string fname, Parameters &p) {
  std::ifstream param_file(fname);
  if(!param_file.is_open()) {
    std::cout << "Unable to open file." << std::endl;
    exit(2);
  }
  json j;
  param_file >> j;
  j.at("devID").get_to(p.devID);
  try {
    j.at("verbose").get_to(p.verbose);
  } catch(nlohmann::detail::out_of_range) {
    // nothing
  }

  json j_image = j.at("input_image_info");
  j_image.at("width").get_to(p.width);
  j_image.at("height").get_to(p.height);
  j_image.at("left_margin").get_to(p.left_margin);
  j_image.at("right_margin").get_to(p.right_margin);
  j_image.at("top_margin").get_to(p.top_margin);
  j_image.at("bottom_margin").get_to(p.bottom_margin);

  json j_coord = j.at("coordinate_info");
  j_coord.at("scale_x").get_to(p.scale_x);
  j_coord.at("scale_y").get_to(p.scale_y);
  j_coord.at("scale_z").get_to(p.scale_z);

  json j_param = j.at("HDoG_param");
  j_param.at("depth").get_to(p.depth);
  p.image_size = p.width * p.height * p.depth;
  p.image_size2D = p.width * p.height;
  j_param.at("extra_depth_margin").get_to(p.extra_depth_margin);
  try {
    j.at("cub_tmp_size_factor").get_to(p.cub_tmp_size_factor);
  } catch(nlohmann::detail::out_of_range) {
    // nothing
  }
  p.cub_tmp_bytes = p.image_size * p.cub_tmp_size_factor;

  try {
    j_param.at("min_intensity_skip_NeuN").get_to(p.min_intensity_skip); //new
    j_param.at("min_intensity_skip_Iba1").get_to(p.min_intensity_skip); //new
  } catch(nlohmann::detail::out_of_range) {
    // nothing
  }
  
  j_param.at("min_intensity_skip").get_to(p.min_intensity_skip);
  std::cout << "min_intensity_skip:" << p.min_intensity_skip << std::endl;

  j_param.at("radius_norm").get_to(p.radius_norm);
  j_param.at("min_intensity").get_to(p.min_intensity_truncate); 
  j_param.at("min_intensity_NeuN").get_to(p.min_intensity_truncate_NeuN); //new
  j_param.at("min_intensity_Iba1").get_to(p.min_intensity_truncate_Iba1); //new
  j_param.at("min_intensity_Iba1_left").get_to(p.min_intensity_truncate_Iba1_left); //new
  std::cout << "min_intensity:" << p.min_intensity_truncate << std::endl;

  j_param.at("center").get_to(p.center);
  j_param.at("offset_right_NeuN").get_to(p.offset_right_NeuN);
  j_param.at("offset_right_Iba1").get_to(p.offset_right_Iba1);
  j_param.at("offset_left_NeuN").get_to(p.offset_left_NeuN);
  j_param.at("offset_left_Iba1").get_to(p.offset_left_Iba1);

  j_param.at("gamma_n").get_to(p.gamma_n);
  j_param.at("sigma_dog").get_to(p.sigma_dog);

  j_param.at("min_size").get_to(p.min_size);
  std::cout << "min_size:" << p.min_size << std::endl;

  try {
    j.at("is_ave_mode").get_to(p.is_ave_mode);
  } catch(nlohmann::detail::out_of_range) {
    // nothing
  }

  json j_stacks = j.at("stacks");
  p.n_stack = j_stacks.size();
  for (json::iterator j_st = j_stacks.begin(); j_st != j_stacks.end(); ++j_st) {
    json j_src = j_st->at("src_paths");
    p.list_stack_length.push_back(j_src.size());
    p.list_src_path.emplace_back();
    j_src.get_to(p.list_src_path.back());
    p.list_dst_path.push_back(j_st->at("dst_path").get<std::string>());
  }


  try {
    json j_search = j.at("parameter_search");
    p.n_search = j_search.size();
    for (json::iterator j_s = j_search.begin(); j_s != j_search.end(); ++j_s) {
      // radius norm
      try {
        json j_search_norm = j_s->at("radius_norm");
        p.list_radius_norm.push_back(j_search_norm.get<param::RadiusNorm>());
      } catch(nlohmann::detail::out_of_range) {
        p.list_radius_norm.push_back(p.radius_norm);
      }
      // sigma dog
      try {
        json j_search_dog = j_s->at("sigma_dog");
        p.list_sigma_dog.push_back(j_search_dog.get<param::SigmaDoG>());
      } catch(nlohmann::detail::out_of_range) {
        p.list_sigma_dog.push_back(p.sigma_dog);
      }
    }
  } catch(nlohmann::detail::out_of_range) {
    p.n_search = 0;
  }

  return;
}

/* Binary Images */
void loadImage(const std::string fname, unsigned short *h_img, const int image_size2D, const int width, const int height) {
  //std::cout << "loadImage(" << fname << ")" << std::endl;
    FILE *f;
 
    std:: cout<< fname<<"\n";

    //std::string fname2= fname;
    //fname2.replace(fname2.find("ex488_em620"),11,"ex642_em720");
    f = fopen(fname.c_str(), "rb");
    if (f == NULL) {
    std::cout << "Unable to open " << fname << std::endl;
    exit(2);
    }
    unsigned short *Ptr = h_img;
    for (int i = 0; i < 2048; ++i)
    {
	fread(Ptr, sizeof(unsigned short), 2048, f);
	fseek(f, 24, SEEK_CUR);
	Ptr += 2048;
    }
    fclose(f);
    //hot_pix = np.array([[1871,1557], [1627,1716],[1401,1730],[946,843], [946,841],[900,1573],[812,184]])
    
    // to hot pixel
    
    int x1 = 1871;
    int y1 = 1557;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 1627;
    y1 = 1716;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 1401;
    y1 = 1730;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 946;
    y1 = 843;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 946;
    y1 = 841;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 900;
    y1 = 1573;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 812;
    y1 = 184;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    
    
     //h_img[2048*1557+1871]= (h_img[2048*1557+1872]+h_img[2048*1557+1870]+h_img[2048*1558+1871]+h_img[2048*1556+1871])/4;
    
}

void loadImage_NeuN(const std::string fname, unsigned short *h_img, const int image_size2D, const int width, const int height) {
  //std::cout << "loadImage(" << fname << ")" << std::endl;
  // verticalに flipが必要。
    FILE *f;
 
    std:: cout<< fname<<"\n";

    std::string fname2= fname;
    fname2.replace(fname2.find("ex488_em620"),11,"ex642_em720");
    // X nameを取得、center以上であれば、left, より小さいなら rightにする
    // 200000.binの部分を置換する、もし、file名のstart end の範囲になければ、補完する
    

    f = fopen(fname2.c_str(), "rb");
    if (f == NULL) {
    std::cout << "Unable to open " << fname2 << std::endl;
    exit(2);
    }
    unsigned short *Ptr = h_img;
    Ptr += 2048*2048; //for flip
    for (int i = 0; i < 2048; ++i)
    { 
  Ptr -= 2048; //for flip
	fread(Ptr, sizeof(unsigned short), 2048, f);
	fseek(f, 24, SEEK_CUR);
  //Ptr +=2048;
    }
    fclose(f);
}

void loadImage_Iba1(const std::string fname, unsigned short *h_img, const int image_size2D, const int width, const int height) {
  //std::cout << "loadImage(" << fname << ")" << std::endl;
    FILE *f;
 
    std:: cout<< fname<<"\n";

    std::string fname2= fname;
    fname2.replace(fname2.find("ex488_em620"),11,"ex592_em620");
    f = fopen(fname2.c_str(), "rb");
    if (f == NULL) {
    std::cout << "Unable to open " << fname2 << std::endl;
    exit(2);
    }
    unsigned short *Ptr = h_img;
    for (int i = 0; i < 2048; ++i)
    {
	fread(Ptr, sizeof(unsigned short), 2048, f);
	fseek(f, 24, SEEK_CUR);
	Ptr += 2048;
    }
    fclose(f);
    
    // to hot pixel
    int x1 = 1871;
    int y1 = 1557;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 1627;
    y1 = 1716;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 1401;
    y1 = 1730;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 946;
    y1 = 843;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 946;
    y1 = 841;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 900;
    y1 = 1573;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
    x1 = 812;
    y1 = 184;
    h_img[2048*x1+y1]= (h_img[2048*(x1+1)+y1]+h_img[2048*(x1-1)+y1]+h_img[2048*x1+y1+1]+h_img[2048*x1+y1-1])/4;
}

template<typename T>
void saveImage(const T* img_result, const std::string fname, const long long size, const std::string mode="wb") {
  //std::cout << "saveImage(" << fname << ")" << std::endl;
  FILE *f;
  if ((f = fopen(fname.c_str(), mode.c_str())) == NULL) {
    std::cout << "Unable to open " << fname << std::endl;
    exit(2);
  }
  fwrite((char *)img_result, sizeof(T), size, f);
  fclose(f);
}

/*  Feature Vector */
struct FeatureVector{
  float centroid_x;
  float centroid_y;
  float centroid_z;
  float structureness;
  float blobness;
  float max_normalized;
  unsigned short size;
  unsigned short padding;
};

/*  Feature Vector */
struct FeatureVector_NeuN{
  float centroid_x;
  float centroid_y;
  float centroid_z;
  float structureness;
  float blobness;
  float max_normalized;
  unsigned short size;
  unsigned short padding;
  float max_normalized_NeuN;
  float max_normalized_Iba1;
  
};


/*  Feature Vector */
struct FeatureVector_NeuN_only{
  float centroid_x;
  float centroid_y;
  float centroid_z;
  float structureness;
  float blobness;
  float max_normalized;
  unsigned short size;
  unsigned short padding;
  float max_normalized_NeuN;
};

void loadFeatureVector(const std::string fname, const Parameters &p,
                       int z0,
                       float offset_x, float offset_y, float offset_z) {
}

void saveFeatureVector
(
 float *h_maxnorm_region,
 unsigned short *h_size_region,
 float *h_eigen_region, float *h_grid_region,
 const std::string fname, int num_regions,
 int depth, int z0, const Parameters &p,
 const std::string mode="wb", float min_maxnorm=0
 ) {
  //std::cout << "saveFeatureVector(" << fname << "," << num_regions << "," << depth << "," << z0 << "," << mode << ")" << std::endl;

  FILE *f;
  if((f = fopen(fname.c_str(), mode.c_str())) == NULL) {
    std::cout << "Unable to open " << fname << std::endl;
    exit(2);
  }

  int count = 0;
  for(int i = 1; i < num_regions; i++) { // skip background(i=0)
    float centroid_x = h_grid_region[i];
    float centroid_y = h_grid_region[p.image_size+i];
    float centroid_z = h_grid_region[2*p.image_size+i];
    //std::cout << centroid_x << "," << centroid_y << "," << centroid_z << std::endl;
    // margins
    if (centroid_z < p.depth_margin || centroid_z >= depth-p.depth_margin) continue;
    if (centroid_x < p.left_margin || centroid_x >= p.width-p.right_margin) continue;
    if (centroid_y < p.top_margin || centroid_y >= p.height-p.bottom_margin) continue;
    // threshold
    if (h_size_region[i] < p.min_size || h_maxnorm_region[i] <= min_maxnorm) continue;

    // save binary FeatureVector struct
    FeatureVector fv = {
      centroid_x,
      centroid_y,
      centroid_z+z0,
      h_eigen_region[i],
      h_eigen_region[p.image_size+i],
      h_maxnorm_region[i],
      h_size_region[i],
      0
    };
    count++;
    fwrite(&fv, sizeof(FeatureVector), 1, f);
  }
  std::cout << "saved count:" << count << "*" << sizeof(FeatureVector) << "=" << count*sizeof(FeatureVector) << std::endl;
  fclose(f);
}

void saveFeatureVector_NeuN
(
 float *h_maxnorm_region,
 float *h_max_normalized_NeuN,
 float *h_max_normalized_Iba1,
 unsigned short *h_size_region,
 float *h_eigen_region, float *h_grid_region,
 const std::string fname, int num_regions,
 int depth, int z0, const Parameters &p,
 const std::string mode="wb", float min_maxnorm=0
 ) {
  //std::cout << "saveFeatureVector(" << fname << "," << num_regions << "," << depth << "," << z0 << "," << mode << ")" << std::endl;

  FILE *f;
  if((f = fopen(fname.c_str(), mode.c_str())) == NULL) {
    std::cout << "Unable to open " << fname << std::endl;
    exit(2);
  }

  int count = 0;
  for(int i = 1; i < num_regions; i++) { // skip background(i=0)
    float centroid_x = h_grid_region[i];
    float centroid_y = h_grid_region[p.image_size+i];
    float centroid_z = h_grid_region[2*p.image_size+i];
    //std::cout << centroid_x << "," << centroid_y << "," << centroid_z << std::endl;
    // margins
    if (centroid_z < p.depth_margin || centroid_z >= depth-p.depth_margin) continue;
    if (centroid_x < p.left_margin || centroid_x >= p.width-p.right_margin) continue;
    if (centroid_y < p.top_margin || centroid_y >= p.height-p.bottom_margin) continue;
    // threshold
    if (h_size_region[i] < p.min_size || h_maxnorm_region[i] <= min_maxnorm) continue;

    // save binary FeatureVector struct
    FeatureVector_NeuN fv = {
      centroid_x,
      centroid_y,
      centroid_z+z0,
      h_eigen_region[i],
      h_eigen_region[p.image_size+i],
      h_maxnorm_region[i],
      h_size_region[i],
      0,
      h_max_normalized_NeuN[i],
      h_max_normalized_Iba1[i],
      
    };
    count++;
    fwrite(&fv, 36, 1, f);
    //fwrite(&fv, sizeof(FeatureVector_NeuN), 1, f);
  }
  std::cout << "saved count:" << count << "*" << sizeof(FeatureVector_NeuN) << "=" << count*sizeof(FeatureVector_NeuN) << std::endl;
  fclose(f);
}


void saveFeatureVector_NeuN_only
(
 float *h_maxnorm_region,
 float *h_max_normalized_NeuN,
 unsigned short *h_size_region,
 float *h_eigen_region, float *h_grid_region,
 const std::string fname, int num_regions,
 int depth, int z0, const Parameters &p,
 const std::string mode="wb", float min_maxnorm=0
 ) {
  //std::cout << "saveFeatureVector(" << fname << "," << num_regions << "," << depth << "," << z0 << "," << mode << ")" << std::endl;

  FILE *f;
  if((f = fopen(fname.c_str(), mode.c_str())) == NULL) {
    std::cout << "Unable to open " << fname << std::endl;
    exit(2);
  }

  int count = 0;
  for(int i = 1; i < num_regions; i++) { // skip background(i=0)
    float centroid_x = h_grid_region[i];
    float centroid_y = h_grid_region[p.image_size+i];
    float centroid_z = h_grid_region[2*p.image_size+i];
    //std::cout << centroid_x << "," << centroid_y << "," << centroid_z << std::endl;
    // margins
    if (centroid_z < p.depth_margin || centroid_z >= depth-p.depth_margin) continue;
    if (centroid_x < p.left_margin || centroid_x >= p.width-p.right_margin) continue;
    if (centroid_y < p.top_margin || centroid_y >= p.height-p.bottom_margin) continue;
    // threshold
    if (h_size_region[i] < p.min_size || h_maxnorm_region[i] <= min_maxnorm) continue;

    // save binary FeatureVector struct
    FeatureVector_NeuN_only fv = {
      centroid_x,
      centroid_y,
      centroid_z+z0,
      h_eigen_region[i],
      h_eigen_region[p.image_size+i],
      h_maxnorm_region[i],
      h_size_region[i],
      0,
      h_max_normalized_NeuN[i],
      
    };
    count++;
    fwrite(&fv, 32, 1, f);
    //fwrite(&fv, sizeof(FeatureVector_NeuN), 1, f);
  }
  std::cout << "saved count:" << count << "*" << sizeof(FeatureVector_NeuN_only) << "=" << count*sizeof(FeatureVector_NeuN_only) << std::endl;
  fclose(f);
}

#endif
