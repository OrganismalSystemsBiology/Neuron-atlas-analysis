#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <string>
#include <chrono>
#include "DoG3DFilter.cuh"
#include "Hessian3DFilter_element.cuh"
#include "CCL3D.cuh"
#include "Erosion3DFilter.cuh"
#include "RegionalFeatures.cuh"
#include "Eigenvalues.cuh"
#include "NormalizeFilter.cuh"
#include "utils_NeuN_ver3.h"
#include <sys/stat.h> // for stat()



Parameters p;

// host memory
unsigned short *h_img = NULL;
int *h_labels = NULL;
int *h_labels_region = NULL;
float *h_norm = NULL; //new
unsigned short *h_img_NeuN = NULL; //new
unsigned short *h_img_Iba1 = NULL; //new
float *h_norm_NeuN = NULL; //new
float *h_norm_Iba1 = NULL; //new
unsigned short *h_size_region = NULL;
float *h_maxnorm_region = NULL;
float *h_eigen_region = NULL;
float *h_grid_region = NULL;
int h_num_regions;
float *h_maxnorm_region_NeuN = NULL; //new
float *h_maxnorm_region_Iba1 = NULL; //new

// device memory
unsigned short *d_img = NULL;
unsigned short *d_img_NeuN = NULL; //new
unsigned short *d_img_Iba1 = NULL; //new
float *d_hessian = NULL;
char *d_hessian_pd = NULL;
int *d_labels = NULL;
int *d_labels_tmp = NULL;
float *d_hessian_tmp = NULL;
int *d_labels_region = NULL;
int *d_labels_region_NeuN = NULL; //new
int *d_labels_region_Iba1 = NULL; //new
unsigned short *d_size_region = NULL;
float *d_eigen_region = NULL;
int* d_num_regions = NULL;
void *d_cub_tmp = NULL;


// recycled device memory
float *d_temp1 = NULL;
float *d_temp2 = NULL;
float *d_norm = NULL;
float *d_dog = NULL;
float *d_hessian_region = NULL;
float *d_grid = NULL;
float *d_grid_region = NULL;
float *d_grid_tmp = NULL;
float *d_norm_tmp = NULL;
float *d_maxnorm_region = NULL;
float *d_maxnorm_region_NeuN = NULL; //new
float *d_maxnorm_region_Iba1 = NULL; //new
unsigned short *d_img_tmp = NULL;
unsigned short *d_erosion_tmp1 = NULL;
unsigned short *d_erosion_tmp2 = NULL;
float *d_erosion_l = NULL;
float *d_dilation_l = NULL;
float *d_norm_NeuN = NULL; //new
float *d_norm_Iba1 = NULL; //new

void initializeGPU() {
  // cuda device init
  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;
  cudaSetDevice(p.devID);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, p.devID));
  //checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

  printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
         deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
  cudaFree(0); // to measure cudaMalloc correctly

  // allocating page-locked host memory
  // h_img: WriteCombined is fast for host->device only
  checkCudaErrors(cudaHostAlloc(&h_img, p.image_size*sizeof(unsigned short), cudaHostAllocWriteCombined));
  checkCudaErrors(cudaHostAlloc(&h_img_NeuN, p.image_size*sizeof(unsigned short), cudaHostAllocWriteCombined)); //New
  checkCudaErrors(cudaHostAlloc(&h_img_Iba1, p.image_size*sizeof(unsigned short), cudaHostAllocWriteCombined)); //New
  checkCudaErrors(cudaHostAlloc(&h_labels, p.image_size*sizeof(int), 0));
  checkCudaErrors(cudaHostAlloc(&h_labels_region, p.image_size*sizeof(int), 0));
  checkCudaErrors(cudaHostAlloc(&h_maxnorm_region, p.image_size*sizeof(float), 0));
  checkCudaErrors(cudaHostAlloc(&h_maxnorm_region_NeuN, p.image_size*sizeof(float), 0)); //new
  checkCudaErrors(cudaHostAlloc(&h_maxnorm_region_Iba1, p.image_size*sizeof(float), 0)); //new
  checkCudaErrors(cudaHostAlloc(&h_size_region, p.image_size*sizeof(unsigned short), 0));
  checkCudaErrors(cudaHostAlloc(&h_eigen_region, 2*p.image_size*sizeof(float), 0));
  checkCudaErrors(cudaHostAlloc(&h_grid_region, 3*p.image_size*sizeof(float), 0));
  //checkCudaErrors(cudaHostAlloc(&h_norm, p.image_size*sizeof(float), 0)); //new 画像出力のために一時的に。
  //checkCudaErrors(cudaHostAlloc(&h_norm_NeuN, p.image_size*sizeof(float), 0)); //new 画像出力のために一時的に。
  //checkCudaErrors(cudaHostAlloc(&h_norm_Iba1, p.image_size*sizeof(float), 0)); //new 画像出力のために一時的に。

  // prepare device memory
  checkCudaErrors(cudaMalloc((void **)&d_img, (p.image_size*sizeof(unsigned short))));
  checkCudaErrors(cudaMalloc((void **)&d_img_NeuN, (p.image_size*sizeof(unsigned short)))); //new
  checkCudaErrors(cudaMalloc((void **)&d_img_Iba1, (p.image_size*sizeof(unsigned short)))); //new
  checkCudaErrors(cudaMalloc((void **)&d_norm, (p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_norm_NeuN, (p.image_size*sizeof(float)))); //new
  checkCudaErrors(cudaMalloc((void **)&d_norm_Iba1, (p.image_size*sizeof(float)))); //new
  checkCudaErrors(cudaMalloc((void **)&d_maxnorm_region_NeuN, (p.image_size*sizeof(float)))); //new  
  checkCudaErrors(cudaMalloc((void **)&d_maxnorm_region_Iba1, (p.image_size*sizeof(float)))); //new  
  checkCudaErrors(cudaMalloc((void **)&d_labels, (p.image_size*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_labels_tmp, (p.image_size*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_labels_region, (p.image_size*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_labels_region_NeuN, (p.image_size*sizeof(int)))); //new
  checkCudaErrors(cudaMalloc((void **)&d_labels_region_Iba1, (p.image_size*sizeof(int)))); //new
  checkCudaErrors(cudaMalloc((void **)&d_hessian, (6*p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian_pd, (p.image_size*sizeof(char))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian_tmp, (p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_size_region, (p.image_size*sizeof(unsigned short))));
  checkCudaErrors(cudaMalloc((void **)&d_eigen_region, (2*p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_num_regions, sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_cub_tmp, p.cub_tmp_bytes));

  // recycle device memory
  d_erosion_tmp1 = reinterpret_cast<unsigned short*>(d_hessian);
  d_erosion_tmp2 = d_erosion_tmp1 + p.image_size;
  d_erosion_l = d_hessian + p.image_size;
  d_dilation_l = d_erosion_l + p.image_size;

  d_temp1 = d_hessian;
  d_temp2 = d_temp1 + p.image_size;
  d_dog = reinterpret_cast<float*>(d_labels_tmp);
  d_hessian_region = d_hessian;

  d_grid = d_hessian;
  d_grid_region = d_hessian + p.image_size;
  d_maxnorm_region = d_hessian + 4*p.image_size;

  d_grid_tmp = d_hessian_tmp;
  d_norm_tmp = d_hessian_tmp;
}

void finalizeGPU() {
  checkCudaErrors(cudaFree(d_img));
  checkCudaErrors(cudaFree(d_img_NeuN)); //new
  checkCudaErrors(cudaFree(d_img_Iba1)); //new
  checkCudaErrors(cudaFree(d_norm));
  checkCudaErrors(cudaFree(d_norm_NeuN)); //new
  checkCudaErrors(cudaFree(d_maxnorm_region_NeuN)); //new
  checkCudaErrors(cudaFree(d_maxnorm_region_Iba1)); //new
  checkCudaErrors(cudaFree(d_labels));
  checkCudaErrors(cudaFree(d_labels_tmp));
  checkCudaErrors(cudaFree(d_labels_region));
  checkCudaErrors(cudaFree(d_labels_region_NeuN)); //new
  checkCudaErrors(cudaFree(d_labels_region_Iba1)); //new
  checkCudaErrors(cudaFree(d_hessian));
  checkCudaErrors(cudaFree(d_hessian_pd));
  checkCudaErrors(cudaFree(d_hessian_tmp));
  checkCudaErrors(cudaFree(d_size_region));
  checkCudaErrors(cudaFree(d_eigen_region));
  checkCudaErrors(cudaFree(d_num_regions));
  checkCudaErrors(cudaFree(d_cub_tmp));

  checkCudaErrors(cudaFreeHost(h_img));
  checkCudaErrors(cudaFreeHost(h_img_NeuN)); //new
  checkCudaErrors(cudaFreeHost(h_img_Iba1)); //new
  //checkCudaErrors(cudaFreeHost(h_norm_NeuN)); //new
  //checkCudaErrors(cudaFreeHost(h_norm_Iba1)); //new
  checkCudaErrors(cudaFreeHost(h_labels));
  checkCudaErrors(cudaFreeHost(h_labels_region));
  checkCudaErrors(cudaFreeHost(h_size_region));
  checkCudaErrors(cudaFreeHost(h_eigen_region));
  checkCudaErrors(cudaFreeHost(h_grid_region));
  //checkCudaErrors(cudaFreeHost(h_norm)); //new
}


void processSubstack(const int i_stack, const int z0, const int z0_NeuN, const int z0_Iba1, const int min_intensity_Iba1, const int Iba1_or_not) {
  std::chrono::system_clock::time_point  start, end;
  double elapsed;

  int depth = std::min(p.depth, p.list_stack_length[i_stack]-z0);
  int image_size = p.image_size2D * depth;
  bool is_last = depth != p.depth;
  std::string* list_src_path = &(p.list_src_path[i_stack])[z0];

  std::string* list_src_path_NeuN = &(p.list_src_path[i_stack])[z0_NeuN];
  std::string* list_src_path_Iba1 = &(p.list_src_path[i_stack])[z0_Iba1];
    
  
  //std::string list_src_path_CCL = *list_src_path;
        
  std::string search_str = "ex488_em620";
  //std::string replace_str = "CCL";
  //size_t pos = list_src_path_CCL.find(search_str);
  //if (pos != std::string::npos) {
  //      list_src_path_CCL.replace(pos, search_str.length(), replace_str);
  //      list_src_path_CCL = list_src_path_CCL.substr(0, list_src_path_CCL.find_last_of('/'));
  //      std::cout << list_src_path_CCL << std::endl;
  // }
    
  //std::string* list_src_path_NeuN = &(p.list_src_path[i_stack])[z0];
  //std::string* list_src_path_Iba1 = &(p.list_src_path[i_stack])[z0];

  //エラー処理必要
  //std::string* list_src_path_NeuN = &(p.list_src_path[i_stack])[z0 + offset_NeuN];

  //std::string* list_src_path_Iba1 = &(p.list_src_path[i_stack])[z0 + offset_Iba1];

  // offsetをずらす　もし、file名のstart end の範囲になければ、補完する

  std::cout << "----------" << std::endl;
  std::cout << "processSubstack(z0=" << z0 << ", depth=" << depth << ")" << std::endl;
  std::cout << "\tfirst_src_path: " << list_src_path[0] << std::endl;
  std::cout << "\tlast_src_path: " << list_src_path[depth-1] << std::endl;

  std::cout << "\tfirst_src_path: " << list_src_path_NeuN[0] << std::endl;
  std::cout << "\tlast_src_path: " << list_src_path_NeuN[depth-1] << std::endl;

  std::cout << "\tfirst_src_path: " << list_src_path_Iba1[0] << std::endl;
  std::cout << "\tlast_src_path: " << list_src_path_Iba1[depth-1] << std::endl;

  // get file name without extension
  std::string z0_filename = list_src_path[0].substr(list_src_path[0].find_last_of("/") + 1);
  z0_filename = z0_filename.substr(0, z0_filename.find_last_of("."));

  // extract 6 digit number from filename
  std::string num_str = z0_filename.substr(z0_filename.length() - 6);

  // convert string to integer
  int z0_filename_num = stoi(num_str);

  if(depth == 0) {
    std::cout << "No img provided for processSubstack()!" << std::endl;
    return;
  }

  start = std::chrono::system_clock::now();
  // load images to host memory
  for(int i=0; i < depth; i++) {
    loadImage(list_src_path[i], &h_img[i*p.image_size2D], p.image_size2D,p.width,p.height);
  }
    // load images to host memory NeuN  new
  for(int i=0; i < depth; i++) {
    loadImage_NeuN(list_src_path_NeuN[i], &h_img_NeuN[i*p.image_size2D], p.image_size2D,p.width,p.height); //new
  }
  

  if (Iba1_or_not == 1){
  for(int i=0; i < depth; i++) {
    loadImage_Iba1(list_src_path_Iba1[i], &h_img_Iba1[i*p.image_size2D], p.image_size2D,p.width,p.height); //new
  }
  }

  



  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << "loadImage took " << elapsed << "msec" << std::endl;

  start = std::chrono::system_clock::now();
  // copy images to device
  checkCudaErrors(cudaMemcpyAsync((unsigned short *)d_img, (unsigned short *)h_img,
                                  image_size*sizeof(unsigned short), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync((unsigned short *)d_img_NeuN, (unsigned short *)h_img_NeuN,
                                  image_size*sizeof(unsigned short), cudaMemcpyHostToDevice)); //new
  if (Iba1_or_not == 1){
    checkCudaErrors(cudaMemcpyAsync((unsigned short *)d_img_Iba1, (unsigned short *)h_img_Iba1,
                                  image_size*sizeof(unsigned short), cudaMemcpyHostToDevice)); //new
  }
  

  // zero-fill if substack is not full
  if(depth < p.depth)
    checkCudaErrors(cudaMemsetAsync((unsigned short *)&d_img[image_size], 0,
                                    (p.image_size-image_size)*sizeof(unsigned short)));
  if(depth < p.depth)
    checkCudaErrors(cudaMemsetAsync((unsigned short *)&d_img_NeuN[image_size], 0,
                                    (p.image_size-image_size)*sizeof(unsigned short))); //new
   if (Iba1_or_not == 1){
  if(depth < p.depth)
    checkCudaErrors(cudaMemsetAsync((unsigned short *)&d_img_Iba1[image_size], 0,
                                    (p.image_size-image_size)*sizeof(unsigned short))); //new
  }

  // launch kernels
  //Normalize3DFilter_skip (d_img, d_norm, d_erosion_tmp1, d_erosion_tmp2,

  if (1==0)
  { // Z方向のgaussian
     // extern "C" void GaussianZFilter
    //(
    //unsigned short *d_img,  unsigned short *d_result,
    //const int sigma_num, const int width, const int height, const int depth
    //);
    // 暫定で、Z gaussian後は、
    GaussianZFilter(d_img, d_img_NeuN, 1 , p.width, p.height, p.depth);// sigma ではなく、半径を代入している。sigma 1なら、これは2になる。
  }


  Normalize3DFilter (d_img, d_norm, d_erosion_tmp1, d_erosion_tmp2,
                    d_erosion_l, d_dilation_l,
                    p.min_intensity_truncate, 
                    p.width, p.height, p.depth,
                    p.radius_norm.large_xy, p.radius_norm.large_z);

  Normalize3DFilter(d_img_NeuN, d_norm_NeuN, d_erosion_tmp1, d_erosion_tmp2,
                    d_erosion_l, d_dilation_l,
                    p.min_intensity_truncate_NeuN, p.width, p.height, p.depth,
                    p.radius_norm.large_xy, p.radius_norm.large_z); //new
   if (Iba1_or_not == 1){
  Normalize3DFilter(d_img_Iba1, d_norm_Iba1, d_erosion_tmp1, d_erosion_tmp2,
                    d_erosion_l, d_dilation_l,
                    min_intensity_Iba1, p.width, p.height, p.depth,
                    p.radius_norm.large_xy, p.radius_norm.large_z); //new
  }
  DoG3DFilter(d_norm, d_temp1, d_temp2, d_dog,
              p.width, p.height, p.depth, p.gamma_n);
  HessianPositiveDefiniteWithElement(d_hessian, d_hessian_pd,
                                     d_dog, d_hessian_tmp,
                                     p.width, p.height, p.depth);
  CCL(d_hessian_pd, d_labels, p.width, p.height, p.depth);

  HessianFeatures(d_labels, d_hessian,
                  d_labels_tmp, d_hessian_tmp,
                  d_labels_region, d_hessian_region,
                  d_cub_tmp, p.cub_tmp_bytes,
                  d_num_regions, p.width, p.height, p.depth);
  // get num_regions in host
  checkCudaErrors(cudaMemcpy(&h_num_regions, d_num_regions, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "num_regions: " << h_num_regions << std::endl;

  Eigenvalues(d_hessian_region, d_eigen_region, h_num_regions, p.image_size);

  // new   sumnormalizeを採用。
  AverageNormalized(d_labels, d_norm, d_labels_tmp, d_norm_tmp,
                  d_labels_region, d_maxnorm_region,
                  d_size_region, 
                  d_cub_tmp, p.cub_tmp_bytes,
                  d_num_regions, h_num_regions, p.width, p.height, p.depth);

 // if (!p.is_ave_mode) {
   // MaxNormalized(d_labels, d_norm, d_labels_tmp, d_norm_tmp,
   //               d_labels_region, d_maxnorm_region,
   //               d_cub_tmp, p.cub_tmp_bytes,
  //                d_num_regions, p.width, p.height, p.depth);
  //} else {
  //  SumNormalized(d_labels, d_norm, d_labels_tmp, d_norm_tmp,
   //               d_labels_region, d_maxnorm_region,
   //               d_cub_tmp, p.cub_tmp_bytes,
   //               d_num_regions, p.width, p.height, p.depth);
  //}

  RegionalSizeAndCentroid(d_labels, d_grid,
                          d_labels_tmp, d_grid_tmp,
                          d_labels_region, d_size_region, d_grid_region,
                          d_cub_tmp, p.cub_tmp_bytes,
                          d_num_regions, h_num_regions,
                          p.width, p.height, p.depth);

  // new   sumnormalizeを採用。　　NeuN 
  AverageNormalized(d_labels, d_norm_NeuN, d_labels_tmp, d_norm_tmp,
                  d_labels_region_NeuN, d_maxnorm_region_NeuN,
                  d_size_region, 
                  d_cub_tmp, p.cub_tmp_bytes,
                  d_num_regions, h_num_regions, p.width, p.height, p.depth);
  if (Iba1_or_not == 1){
  AverageNormalized(d_labels, d_norm_Iba1, d_labels_tmp, d_norm_tmp,
                  d_labels_region_Iba1, d_maxnorm_region_Iba1,
                  d_size_region, 
                  d_cub_tmp, p.cub_tmp_bytes,
                  d_num_regions, h_num_regions, p.width, p.height, p.depth);
  }
    


  // NeuN new

  // download to check the result
  //checkCudaErrors(cudaMemcpyAsync((char *)h_labels, (char *)d_labels, image_size*sizeof(int), cudaMemcpyDeviceToHost));
  // download only number of regions
  checkCudaErrors(cudaMemcpyAsync((char *)h_size_region, (char *)d_size_region, h_num_regions*sizeof(unsigned short), cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaMemcpyAsync((char *)h_labels_region, (char *)d_labels_region, h_num_regions*sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync((char *)h_maxnorm_region, (char *)d_maxnorm_region, h_num_regions*sizeof(float), cudaMemcpyDeviceToHost));
   //checkCudaErrors(cudaMemcpyAsync((char *)h_labels_region, (char *)d_labels_region, h_num_regions*sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync((char *)h_maxnorm_region_NeuN, (char *)d_maxnorm_region_NeuN, h_num_regions*sizeof(float), cudaMemcpyDeviceToHost)); //new
   //checkCudaErrors(cudaMemcpyAsync((char *)h_labels_region, (char *)d_labels_region, h_num_regions*sizeof(int), cudaMemcpyDeviceToHost)); 
   if (Iba1_or_not == 1){
  checkCudaErrors(cudaMemcpyAsync((char *)h_maxnorm_region_Iba1, (char *)d_maxnorm_region_Iba1, h_num_regions*sizeof(float), cudaMemcpyDeviceToHost)); //new
   }
 // download for analysis for normalized image 保存 一時的
 //checkCudaErrors(cudaMemcpyAsync((char *)h_img, (char *) d_img, image_size*sizeof(unsigned short), cudaMemcpyDeviceToHost)); //New
 // checkCudaErrors(cudaMemcpyAsync((char *)h_img_NeuN, (char *) d_img_NeuN , image_size*sizeof(unsigned short), cudaMemcpyDeviceToHost)); //New
  //checkCudaErrors(cudaMemcpyAsync((char *)h_norm, (char *)d_norm, image_size*sizeof(float), cudaMemcpyDeviceToHost)); //new
  //checkCudaErrors(cudaMemcpyAsync((char *)h_norm, (char *)d_norm, image_size*sizeof(float), cudaMemcpyDeviceToHost)); //new
 // download for analysis for normalized image 保存　一時的
 // checkCudaErrors(cudaMemcpyAsync((char *)h_norm_NeuN, (char *)d_norm_NeuN, image_size*sizeof(float), cudaMemcpyDeviceToHost)); //new
// download for analysis for normalized image 保存　一時的
  //checkCudaErrors(cudaMemcpyAsync((char *)h_norm_Iba1, (char *)d_norm_Iba1, image_size*sizeof(float), cudaMemcpyDeviceToHost)); //new

  for(int i = 0; i < 3; i++) {
    checkCudaErrors(cudaMemcpyAsync((char *)(h_grid_region+i*p.image_size), (char *)(d_grid_region+i*p.image_size), h_num_regions*sizeof(float), cudaMemcpyDeviceToHost));
  }
  for(int i = 0; i < 2; i++) {
    checkCudaErrors(cudaMemcpyAsync((char *)(h_eigen_region+i*p.image_size), (char *)(d_eigen_region+i*p.image_size), h_num_regions*sizeof(float), cudaMemcpyDeviceToHost));
  }

  //barrier (without this, next substack kernel call would overlap!!)
  checkCudaErrors(cudaDeviceSynchronize());
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << "GPU memcpy & kernel execution took " << elapsed << "msec" << std::endl;


if (1==0)
{   //  label image 保存
    char num_char[1000];

    if (1==1)
    {
      char num_char[1000];
      for(int i = 0; i <16; i++) {
      std::sprintf(num_char, "/export3/Imaging/Axial/Neurorology/Nuclear_test/%08d.bin", z0+8+i);
      std::cout<<"z"<<z0+8+i<<std::endl;
      std::cout<<num_char<<"////////////"<<std::endl;
      FILE *fp;
      fp = fopen(num_char,"wb");
      fwrite(h_img_NeuN+p.image_size2D,sizeof(unsigned short),p.image_size2D*depth,fp);
      fclose(fp);
    }


    for(int i = 0; i <16; i++) {
      std::sprintf(num_char, "/export3/Imaging/Axial/Neurorology/Nuclear_test_src/%08d.bin", z0+8+i);
      std::cout<<"z"<<z0+8+i<<std::endl;
      std::cout<<num_char<<"////////////"<<std::endl;
      FILE *fp;
      fp = fopen(num_char,"wb");
      fwrite(h_img+p.image_size2D,sizeof(unsigned short),p.image_size2D*depth,fp);
      fclose(fp);
    }

    }
    
    if (1==0)
    {
    for(int i = 0; i <32; i++) {
   // std::sprintf(num_char, list_src_path_CCL.substr(0, list_src_path[0].length() - 18).c_str() + "/%%08d.bin", z0+8+i);
    //std::sprintf(num_char, (std::string(list_src_path_CCL.substr(0, list_src_path[0].length() - 18)) + "/"+"%06d.bin").c_str(), 200000+z0+8+i);
    //std::sprintf(num_char, (std::string(list_src_path_CCL.substr(0, list_src_path[0].length() - 18)) + "/" + std::to_string(z0_filename_num ) + "_" +std::to_string(z0_filename_num  + i) + ".bin").c_str());
    
    std::cout<<"z"<<z0_filename_num + z0 + i<<std::endl;
    std::cout<<num_char<<"////////////"<<std::endl;
    FILE *fp;
    fp = fopen(num_char,"wb");
    fwrite(h_labels+p.image_size2D*(i),sizeof(int),p.image_size2D,fp);
    fclose(fp);
    }
    }
    
   //std::sprintf(num_char, (std::string(list_src_path_CCL.substr(0, list_src_path[0].length() - 18)) + "/" + std::to_string(200000 + z0) + "_" + std::to_string(depth) + ".bin").c_str());
   // FILE *fp;
    //fp = fopen(num_char,"wb");
    //fwrite(h_norm+p.image_size2D*(8),sizeof(float),p.image_size2D,fp);
    //fclose(fp);
}
if (1==0)
{   //  Normalized image 保存
    char num_char[1000];
    for(int i = 0; i <16; i++) {
    std::sprintf(num_char, "/export3/Imaging/Axial/Neurorology/Nuclear_test/%08d.bin", z0+8+i);
    std::cout<<"z"<<z0+8+i<<std::endl;
    std::cout<<num_char<<"////////////"<<std::endl;
    FILE *fp;
    fp = fopen(num_char,"wb");
    fwrite(h_norm+p.image_size2D,sizeof(float),p.image_size2D*depth,fp);
    fclose(fp);
    }
}
if (1==0)
{   //  Normalized image 保存 NeuN
    char num_char[1000];
    for(int i = 0; i <16; i++) {
    std::sprintf(num_char, "/export3/Imaging/Axial/Neurorology/NeuN_test/%08d.bin", z0+8+i);
    std::cout<<"z"<<z0+8+i<<std::endl;
    std::cout<<num_char<<"////////////"<<std::endl;
    FILE *fp2;
    fp2 = fopen(num_char,"wb");
    fwrite(h_norm_NeuN+p.image_size2D*(8+i),sizeof(float),p.image_size2D,fp2);
    fclose(fp2);
    }

if (1==0)
{   //  Normalized image 保存 Iba1
    char num_char[1000];
    for(int i = 0; i <16; i++) {
    std::sprintf(num_char, "/export3/Imaging/Axial/Neurorology/Iba1_test/%08d.bin", z0+8+i);
    std::cout<<"z"<<z0+8+i<<std::endl;
    std::cout<<num_char<<"////////////"<<std::endl;
    FILE *fp3;
    fp3 = fopen(num_char,"wb");
    fwrite(h_norm_Iba1+p.image_size2D*(8+i),sizeof(float),p.image_size2D,fp3);
    fclose(fp3);
    }
}


}



  start = std::chrono::system_clock::now();
  //saveFeatureVector(h_maxnorm_region,
                    //h_size_region,
                    //h_eigen_region, h_grid_region,
                    //p.list_dst_path[i_stack],
                    //h_num_regions, depth, z0, p, "ab");
   if (Iba1_or_not == 1){
  saveFeatureVector_NeuN(h_maxnorm_region, h_maxnorm_region_NeuN, h_maxnorm_region_Iba1,
                    h_size_region,
                    h_eigen_region, h_grid_region,
                    p.list_dst_path[i_stack],
                   h_num_regions, depth, z0, p, "ab"); //消した。
   }

    if (Iba1_or_not == 0){
   saveFeatureVector_NeuN(h_maxnorm_region, h_maxnorm_region_NeuN, h_maxnorm_region_NeuN,
                    h_size_region,
                    h_eigen_region, h_grid_region,
                    p.list_dst_path[i_stack],
                   h_num_regions, depth, z0, p, "ab"); //消した。
    }


  //書いた
  //saveFeatureVector_NeuN(h_maxnorm_region, h_maxnorm_region, h_maxnorm_region,
  //                  h_size_region,
  //                  h_eigen_region, h_grid_region,
  //                  p.list_dst_path[i_stack],
  //                  h_num_regions, depth, z0, p, "ab");

  if(!is_last) {
    // middle of the stack
    std::cout << "[save] middle of the stack : "
              << z0 + p.depth_margin << " - " << z0 + depth - p.depth_margin - 1
              << " (" << depth - 2*p.depth_margin << ")" << std::endl;
    //saveImage(h_labels+p.depth_margin*p.image_size2D, dst_path, p.image_size2D*(depth-2*p.depth_margin), "ab");
  } else {
    // end of the stack
    std::cout << "[save] end of the stack : "
              << z0 + p.depth_margin << " - " << z0 + depth - 1
              << " (" << depth - p.depth_margin << ")" << std::endl;
    //saveImage(h_labels+p.depth_margin*p.image_size2D, dst_path, p.image_size2D*(depth-p.depth_margin), "ab");
  }
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << "saveImage took " << elapsed << "msec" << std::endl;

}


int main(int argc, char** argv) {
  std::chrono::system_clock::time_point  start, end;
  double elapsed;

  if( argc != 2 ) {
    std::cout << "Usage:" << std::endl;
    std::cout << argv[0] << " PARAM_FILE" << std::endl;
    exit(2);
  }

  loadParamFile(argv[1], p);
  std::cout << "number of stacks: " << p.n_stack << std::endl;

  start = std::chrono::system_clock::now();
  initializeGPU(); // GPU device ID
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << "Initialization took " << elapsed << "msec" << std::endl;

  // create DoG kernel, and set depth_margin
  int radius_dog = initGaussian3DKernel(p.sigma_dog.small_xy,
                                        p.sigma_dog.small_z,
                                        p.sigma_dog.large_xy,
                                        p.sigma_dog.large_z);
  p.depth_margin = radius_dog + p.extra_depth_margin;

  std::cout << "sigma_xy1: " << p.sigma_dog.small_xy << "\t"
            << "sigma_z1: "  << p.sigma_dog.small_z << "\t"
            << "sigma_xy2: " << p.sigma_dog.large_xy << "\t"
            << "sigma_z2: "  << p.sigma_dog.large_z << std::endl;
  std::cout << "radius_large_xy: " << p.radius_norm.large_xy << "\t"
            << "radius_large_z: "  << p.radius_norm.large_z <<  std::endl;
  std::cout << "depth_margin: " << p.depth_margin << std::endl;



  int offset_NeuN=0;
  int offset_Iba1=0;
  int z0_NeuN=0;
  int z0_Iba1=0;

  int min_intensity_Iba1 = 0;

  for(int i_stack = 0; i_stack < p.n_stack; i_stack++) {
    if (remove(p.list_dst_path[i_stack].c_str()) == 0) {
      // because result is written in append mode, the file is removed beforehand
      std::cout << "removed previous result" << std::endl;
    } else {
      std::cout << "newly create result" << std::endl;
    }

    // one stack(`length` images) is partitioned into many substacks.
    // substacks(`depth` images) have overlap with neighboring substacks.
    // for each substack, images [0,depth-1] are used, but regions whose
    // centroid_z is in [0,margin-1] or [depth-margin,depth-1] is discarded.
    // Thus, effective z range is [margin,depth-margin-1].
    // For substacks at the end of stack, end margin is not discarded.

    for(int z0 = 0;
        z0 < p.list_stack_length[i_stack] - p.depth_margin;
        z0 += (p.depth-2*p.depth_margin)) {
      // index z0 indicate the top of substack in stack-wide coordinate


  std::string* list_src_path = &(p.list_src_path[i_stack])[z0]; //i_stack
        
        
  // CCLのsave用のfolderを作製。
  std::string list_src_path_CCL = *list_src_path;
        
  std::string search_str = "ex488_em620";
  std::string replace_str = "CCL";
  size_t pos = list_src_path_CCL.find(search_str);
  if (pos != std::string::npos) {
        list_src_path_CCL.replace(pos, search_str.length(), replace_str);
        list_src_path_CCL = list_src_path_CCL.substr(0, list_src_path_CCL.find_last_of('/'));
        std::cout << list_src_path_CCL << std::endl;
   }
        
  // "/export3/Imaging/Axial/Neurorology/test3/CCL_RV/172100/172100_243800/"　のうちの CCL_FWまで
        
  std::string parent_folder = list_src_path_CCL.substr(0, list_src_path[0].length() - 40); 
  struct stat buffer;
  if (stat(parent_folder.c_str(), &buffer) != 0) 
  {
     //std::string command = "mkdir " + parent_folder;
     //system(command.c_str());
  }
        
  std::string parent_folder2 = list_src_path_CCL.substr(0, list_src_path[0].length() - 33); 
  //struct stat buffer;
  if (stat(parent_folder2.c_str(), &buffer) != 0) 
  {
     //std::string command = "mkdir " + parent_folder2;
     //system(command.c_str());
  }
        
  std::string parent_folder3 = list_src_path_CCL.substr(0, list_src_path[0].length() - 19); 
 // struct stat buffer;
  if (stat(parent_folder3.c_str(), &buffer) != 0) 
  {
     //std::string command = "mkdir " + parent_folder3;
     //system(command.c_str());
  }


  // X nameを取得、center以上であれば、left, より小さいなら rightにする

  std::string x_name = list_src_path[0].substr(list_src_path[0].length() - 17 , 6); // x name
  std::string FWRV = list_src_path[0].substr(list_src_path[0].length() - 34 , 2); // x name


  struct stat statbuf;
  int iba1_or_not = 1;

  std::string iba1_folder_name = list_src_path[0];
  search_str = "ex488_em620";
  replace_str = "ex592_em620";
  pos = iba1_folder_name.find(search_str);
  if (pos != std::string::npos) {
        iba1_folder_name.replace(pos, search_str.length(), replace_str);
        iba1_folder_name = iba1_folder_name.substr(0, iba1_folder_name.find_last_of('/'));
        std::cout << iba1_folder_name << std::endl;
   }

if (stat(iba1_folder_name.c_str(), &statbuf) != 0) {
    iba1_or_not = 0;
    std::cout << "Iba1 folder does not exist." << std::endl;
}

  int x_name_num = stoi(x_name);
  //std::cout << x_name << std::endl;
  std::cout << "x_name_num " << x_name_num << std::endl;
  if (FWRV=="FW")
  {
    std::cout << "FW" << std::endl;
  if (x_name_num > p.center)//center 18のとき、Left 19,20,Right18,17
  {
    offset_NeuN = p.offset_left_NeuN;
    offset_Iba1 = p.offset_left_Iba1;

    min_intensity_Iba1 = p.min_intensity_truncate_Iba1_left;
    
    std::cout << "select left" << std::endl;
  }
  else
  {
    offset_NeuN = p.offset_right_NeuN;
    offset_Iba1 = p.offset_right_Iba1;

    min_intensity_Iba1 = p.min_intensity_truncate_Iba1;

    std::cout << "select right" << std::endl;
  }

  } 
    if (FWRV=="RV") //center 18のとき、Left 18,17,Right 19, 20
  {
    std::cout << "RV" << std::endl;

  if (x_name_num > p.center) 
  {
    offset_NeuN = p.offset_right_NeuN;
    offset_Iba1 = p.offset_right_Iba1;

    min_intensity_Iba1 = p.min_intensity_truncate_Iba1;

    std::cout << "select right" << std::endl;
    
    

  }
  else
  {
    offset_NeuN = p.offset_left_NeuN;
    offset_Iba1 = p.offset_left_Iba1;

     min_intensity_Iba1 = p.min_intensity_truncate_Iba1_left;

    std::cout << "select left" << std::endl;
  }
  
  }


  std::cout << "selectNeuN " << offset_NeuN << std::endl;
  std::cout << "selectIba1 " << offset_Iba1 << std::endl;
  std::cout << "min_intensity_Iba1 " << min_intensity_Iba1 << std::endl;

  z0_NeuN = z0 + offset_NeuN;
  z0_Iba1 = z0 + offset_Iba1;

  if (z0_NeuN < 0)
  {
      z0_NeuN = z0;
  }
  if (z0_Iba1 < 0)
  {
      z0_Iba1 = z0;
  }

  if (z0_NeuN > p.list_stack_length[i_stack] -  (p.depth) )
  {
      z0_NeuN = z0 ;
  }

    if (z0_Iba1 > p.list_stack_length[i_stack] -  (p.depth) )
  {
      z0_Iba1 = z0 ;
  }
 
  std::cout << "z0_max " << p.list_stack_length[i_stack] -  (p.depth) << std::endl;
  std::cout << "z0_NeuN " << z0_NeuN << std::endl;
  std::cout << "z0_Iba1 " << z0_Iba1 << std::endl;


      //z0_NeuN = z0 + offset_NeuN;
      //z0_Iba1 = z0 + offset_Iba1;

      processSubstack(i_stack, z0, z0_NeuN, z0_Iba1, min_intensity_Iba1, iba1_or_not) ;
    }
  }

  finalizeGPU();
  return 0;
}
