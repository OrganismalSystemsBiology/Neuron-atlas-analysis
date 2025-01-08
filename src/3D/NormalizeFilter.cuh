#ifndef NORMALIZE_FILTER_CUH
#define NORMALIZE_FILTER_CUH


void Normalize3DFilter
(
 unsigned short *d_img, float *d_norm,
 unsigned short *d_erosion_temp1, unsigned short *d_erosion_temp2,
 float *d_erosion_l, float *d_dilation_l, float min_erosion_l,
 const int width, const int height, const int depth,
 const int radius_large_xy, const int radius_large_z
 );

void Normalize3DFilter_skip
(
 unsigned short *d_img, float *d_norm,
 unsigned short *d_erosion_temp1, unsigned short *d_erosion_temp2,
 float *d_erosion_l, float *d_dilation_l, float min_erosion_l, 
 const int width, const int height, const int depth,
 const int radius_large_xy, const int radius_large_z
 );

 void DilationLarge3DFilter //new
(
 unsigned short *d_img, unsigned short *d_temp,
 unsigned short *d_result,
 int w, int h, int d, int radius_xy, int radius_z
 );


#endif
