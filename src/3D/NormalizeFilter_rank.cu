/*
 * NormalizeFilter_rank.cu
 *
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "NormalizeFilter_rank.cuh"
#include <chrono>
#include <fstream> 

/* --- for large size --- */
// width = 2560 = 5 * 2^9  = divisible by 5*128 = 5 * 2^7
// radius = 200 <= 128 * 2 = 256
// smem_size(minmax) = 2*128*(5+2*2)*2*2 = 9216 < 49152
// smem_size(uniform) = 4*128*(5+2*2)*2*2 = 18432 < 49152
constexpr int L_ROW_BLOCKDIM_X = 64;
constexpr int L_ROW_BLOCKDIM_Y = 2;
constexpr int L_ROW_BLOCKDIM_Z = 2;
constexpr int L_ROW_RESULT_STEPS = 4;
constexpr int L_ROW_HALO_STEPS = 4;

// height = 2160 = 3^3 * 5 * 2^4  = divisible by 144*5 = 3^2 * 5 * 2^4
// radius = 200 <= 144 * 2 = 288
// smem_size(minmax) = 2*144*(5+2*2)*2*2 = 10368 < 49152
// smem_size(uniform) = 4*144*(5+2*2)*2*2 = 20736 < 49152
constexpr int L_COL_BLOCKDIM_X = 2;
constexpr int L_COL_BLOCKDIM_Y = 64;
constexpr int L_COL_BLOCKDIM_Z = 2;
constexpr int L_COL_RESULT_STEPS = 4;
constexpr int L_COL_HALO_STEPS = 4;

// depth = 32 = 2^5  = divisible by 16*2 = 2^5
// radius = 30 < 16 * 2 = 32
// smem_size(minmax) = 2*16*(2+2*4)*8*8 = 20480 < 49152
// smem_size(uniform) = 4*16*(2+2*4)*8*8 = 40960 < 49152
constexpr int L_LAY_BLOCKDIM_X = 8;
constexpr int L_LAY_BLOCKDIM_Y = 8;
constexpr int L_LAY_BLOCKDIM_Z = 16;
constexpr int L_LAY_RESULT_STEPS = 2;
constexpr int L_LAY_HALO_STEPS = 2;

constexpr int NORM_BLOCKDIM_X = 8;
constexpr int NORM_BLOCKDIM_Y = 8;
constexpr int NORM_BLOCKDIM_Z = 8;



/*
 * MinMax(Erosion or Dilation) Filter
 */
template <int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
          int RESULT_STEPS, int HALO_STEPS>
__global__ void Rank_Rows3DKernel(
    float *d_dst, unsigned short *d_src,
    int w, int h, int d,
    int kernel_radius)
{
  __shared__ unsigned short smem[BLOCKDIM_Z][BLOCKDIM_Y][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_X];
  unsigned short *smem_thread = smem[threadIdx.z][threadIdx.y];

  // Offset to the left halo edge
  const int baseX = (blockIdx.x * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_X + threadIdx.x;
  const int baseY = blockIdx.y * BLOCKDIM_Y + threadIdx.y;
  const int baseZ = blockIdx.z * BLOCKDIM_Z + threadIdx.z;

  d_src += (baseZ * h + baseY) * w + baseX;
  d_dst += (baseZ * h + baseY) * w + baseX;

// Load main data
#pragma unroll
  for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
  {
    smem_thread[threadIdx.x + i * BLOCKDIM_X] = d_src[i * BLOCKDIM_X];
  }

// Load left halo (nearest constant border)
#pragma unroll
  for (int i = 0; i < HALO_STEPS; i++)
  {
    smem_thread[threadIdx.x + i * BLOCKDIM_X] = (baseX + i * BLOCKDIM_X >= 0) ? d_src[i * BLOCKDIM_X] : d_src[-baseX];
  }

// Load right halo (nearest constant border)
#pragma unroll
  for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++)
  {
    smem_thread[threadIdx.x + i * BLOCKDIM_X] = (baseX + i * BLOCKDIM_X < w) ? d_src[i * BLOCKDIM_X] : d_src[w - 1 - baseX];
  }

  // Compute and store results
  __syncthreads();
#pragma unroll
  for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
  {
    unsigned short *smem_kern = &smem_thread[threadIdx.x + i * BLOCKDIM_X - kernel_radius];
    float val = (float) smem_kern[0];

    float rank = 0.0;
    float ties = 1.0;

    for (int j = 1; j <= 2 * kernel_radius; j++)
    {
      if ((smem_kern[j] < val))
      {
        rank++;
      }
      else if (smem_kern[j] == val)
      {
        ties++;
      }
    }
    rank += (ties - 1) * 0.5;
    d_dst[i * BLOCKDIM_X] = rank;
  }
}

template <int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
          int RESULT_STEPS, int HALO_STEPS>
void Rank_Rows3D(
    float *d_dst, unsigned short *d_src,
    int w, int h, int d,
    int kernel_radius)
{
  assert(BLOCKDIM_X * HALO_STEPS >= kernel_radius);
  assert(w % (RESULT_STEPS * BLOCKDIM_X) == 0);
  assert(h % BLOCKDIM_Y == 0);
  assert(d % BLOCKDIM_Z == 0);

  dim3 blocks(w / (RESULT_STEPS * BLOCKDIM_X), h / BLOCKDIM_Y, d / BLOCKDIM_Z);
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);

  Rank_Rows3DKernel<BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z, RESULT_STEPS, HALO_STEPS><<<blocks, threads>>>(
      d_dst, d_src, w, h, d, kernel_radius);
  getLastCudaError("Rank_Rows3DKernel_rank() execution failed\n");
  // checkCudaErrors(cudaDeviceSynchronize());
}

template <int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
          int RESULT_STEPS, int HALO_STEPS>
__global__ void Rank_Columns3DKernel(
    float *d_dst, float *d_src,
    int w, int h, int d,
    int kernel_radius)
{
  __shared__ unsigned short smem[BLOCKDIM_Z][BLOCKDIM_X][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_Y + 1];
  unsigned short *smem_thread = smem[threadIdx.z][threadIdx.x];

  // Offset to the upper halo edge
  const int baseX = blockIdx.x * BLOCKDIM_X + threadIdx.x;
  const int baseY = (blockIdx.y * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_Y + threadIdx.y;
  const int baseZ = blockIdx.z * BLOCKDIM_Z + threadIdx.z;

  d_src += (baseZ * h + baseY) * w + baseX;
  d_dst += (baseZ * h + baseY) * w + baseX;

// Main data
#pragma unroll
  for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
  {
    smem_thread[threadIdx.y + i * BLOCKDIM_Y] = d_src[i * BLOCKDIM_Y * w];
  }

// Upper halo (nearest constant border)
#pragma unroll
  for (int i = 0; i < HALO_STEPS; i++)
  {
    smem_thread[threadIdx.y + i * BLOCKDIM_Y] = (baseY + i * BLOCKDIM_Y >= 0) ? d_src[i * BLOCKDIM_Y * w] : d_src[-baseY * w];
  }

// Lower halo (nearest constant border)
#pragma unroll
  for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++)
  {
    smem_thread[threadIdx.y + i * BLOCKDIM_Y] = (baseY + i * BLOCKDIM_Y < h) ? d_src[i * BLOCKDIM_Y * w] : d_src[(h - 1 - baseY) * w];
  }

  // Compute and store results
  __syncthreads();
#pragma unroll
  for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
  {
    unsigned short *smem_kern = &smem_thread[threadIdx.y + i * BLOCKDIM_Y - kernel_radius];
    //float val = smem_kern[0];
    float val = 0;
    //float rank = 0.0;
    //float ties = 1.0;

    // #pragma unroll
    for (int j = 1; j <= 2 * kernel_radius; j++)
    {
        val += smem_kern[j];
    }

    //rank += (ties - 1) * 0.5;
    d_dst[i * BLOCKDIM_Y * w] = val ;
  }
}

template <int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
          int RESULT_STEPS, int HALO_STEPS>
void Rank_Columns3D(
    float *d_dst, float *d_src,
    int w, int h, int d, int kernel_radius)
{
  assert(BLOCKDIM_Y * HALO_STEPS >= kernel_radius);
  assert(w % BLOCKDIM_X == 0);
  assert(h % (RESULT_STEPS * BLOCKDIM_Y) == 0);
  assert(d % BLOCKDIM_Z == 0);

  dim3 blocks(w / BLOCKDIM_X, h / (RESULT_STEPS * BLOCKDIM_Y), d / BLOCKDIM_Z);
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);

  Rank_Columns3DKernel<BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z, RESULT_STEPS, HALO_STEPS><<<blocks, threads>>>(
      d_dst, d_src, w, h, d, kernel_radius);
  getLastCudaError("Rank_Columns3DKernel() execution failed\n");
  // checkCudaErrors(cudaDeviceSynchronize());
}

template <int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
          int RESULT_STEPS, int HALO_STEPS>
__global__ void Rank_Layers3DKernel(
    float *d_dst, float *d_src,
    int w, int h, int d,
    int kernel_radius)
{
  __shared__ unsigned short smem[BLOCKDIM_X][BLOCKDIM_Y][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_Z + 1];
  unsigned short *smem_thread = smem[threadIdx.x][threadIdx.y];

  // Offset to the upper halo edge
  const int baseX = blockIdx.x * BLOCKDIM_X + threadIdx.x;
  const int baseY = blockIdx.y * BLOCKDIM_Y + threadIdx.y;
  const int baseZ = (blockIdx.z * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_Z + threadIdx.z;

  d_src += (baseZ * h + baseY) * w + baseX;
  d_dst += (baseZ * h + baseY) * w + baseX;

  const int pitch = w * h;

// Main data
#pragma unroll
  for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
  {
    smem_thread[threadIdx.z + i * BLOCKDIM_Z] = d_src[i * BLOCKDIM_Z * pitch];
  }

// Upper halo (nearest constant border)
#pragma unroll
  for (int i = 0; i < HALO_STEPS; i++)
  {
    smem_thread[threadIdx.z + i * BLOCKDIM_Z] = (baseZ + i * BLOCKDIM_Z >= 0) ? d_src[i * BLOCKDIM_Z * pitch] : d_src[-baseZ * w * h];
  }

// Lower halo (nearest constant border)
#pragma unroll
  for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++)
  {
    smem_thread[threadIdx.z + i * BLOCKDIM_Z] = (baseZ + i * BLOCKDIM_Z < d) ? d_src[i * BLOCKDIM_Z * pitch] : d_src[(d - 1 - baseZ) * w * h];
  }

  // Compute and store results
  __syncthreads();
#pragma unroll
  for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
  {
    unsigned short *smem_kern = &smem_thread[threadIdx.z + i * BLOCKDIM_Z - kernel_radius];
    //float val = smem_kern[0];
    float val = 0;
    //float rank = 0.0;
    //float ties = 1.0;

    // #pragma unroll
    for (int j = 1; j <= 2 * kernel_radius; j++)
    {
        val += smem_kern[j];
    }

    //rank += (ties - 1) * 0.5;
    d_dst[i * BLOCKDIM_Z * pitch] = val;
  }
}

template <int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
          int RESULT_STEPS, int HALO_STEPS>
void Rank_Layers3D(
    float *d_dst, float *d_src,
    int w, int h, int d,
    int kernel_radius)
{
  assert(BLOCKDIM_Z * HALO_STEPS >= kernel_radius);
  assert(w % BLOCKDIM_X == 0);
  assert(h % BLOCKDIM_Y == 0);
  assert(d % (RESULT_STEPS * BLOCKDIM_Z) == 0);

  dim3 blocks(w / BLOCKDIM_X, h / BLOCKDIM_Y, d / (RESULT_STEPS * BLOCKDIM_Z));
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);

  Rank_Layers3DKernel<BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z, RESULT_STEPS, HALO_STEPS><<<blocks, threads>>>(
      d_dst, d_src, w, h, d, kernel_radius);
  getLastCudaError("Rank_Layers3DKernel() execution failed\n");
  // checkCudaErrors(cudaDeviceSynchronize());
}

/*
 * Define Functions
 */

void Rank_3DFilter(
    unsigned short *d_img, float *d_temp,
    float *d_result,
    int w, int h, int d, int radius_xy, int radius_z)
{
  Rank_Rows3D<L_ROW_BLOCKDIM_X, L_ROW_BLOCKDIM_Y, L_ROW_BLOCKDIM_Z, L_ROW_RESULT_STEPS, L_ROW_HALO_STEPS>(d_result, d_img, w, h, d, radius_xy);
  Rank_Columns3D<L_COL_BLOCKDIM_X, L_COL_BLOCKDIM_Y, L_COL_BLOCKDIM_Z, L_COL_RESULT_STEPS, L_COL_HALO_STEPS>(d_temp, d_result, w, h, d, radius_xy);
  Rank_Layers3D<L_LAY_BLOCKDIM_X, L_LAY_BLOCKDIM_Y, L_LAY_BLOCKDIM_Z, L_LAY_RESULT_STEPS, L_LAY_HALO_STEPS>(d_result, d_temp, w, h, d, radius_z);
}

/*
 * Rank Filter
 */
__global__ void rank_3d_filter_kernel(unsigned short* input, float* output, int width, int height, int depth, float min_intensity) 
{
    const int window_width = 16;
    const int window_height = 16;
    const int window_depth = 4;


    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int z = blockIdx.z * blockDim.z + tz;

    if (x >= width || y >= height || z >= depth) {
        return;
    }

    const int window_size = (window_width * 2 + 1)  * (window_height* 2 + 1) * (window_depth* 2 + 1) ;
    const float center_val = static_cast<float>(input[z * width * height + y * width + x]);

    float rank = 0.0;
    float ties  = 1.0;
    int empty = 0;

    if (center_val  > min_intensity)
    {
    for (int k = - window_depth; k <= window_depth; k ++) {
        for (int j = -window_height; j <= window_height; j ++) {
            for (int i = -window_width; i <=window_width; i ++) {
                if (x+i < 0 || x+i >= width || y+j < 0 || y+j >= height || z+k < 0 || z+k >= depth) {
                    empty++;
                } 
                else {
                    if (static_cast<float>(input[( (z + k) * height + y + j) * width + x + i]) < center_val)
                    {
                        rank++;
                    }

                    else if (static_cast<float>(input[( (z + k) * height + y + j) * width + x + i]) == center_val)
                    {
                        ties++;
                    }

                }

            }
        }
    }
    }
        
    rank += (ties - 1) / 2.0;

    output[z * width * height + y * width + x] = rank / (window_size - empty);
}

void rankFilter3D(
    unsigned short *d_img, float *d_norm,
    unsigned short *d_erosion_temp1, unsigned short *d_erosion_temp2,
    float *d_erosion_l, float *d_dilation_l,
    float min_intensity,
    const int width, const int height, const int depth,
    const int radius_large_xy, const int radius_large_z
)
{
   dim3 dimBlock(16, 16, 1);
   dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, (depth + dimBlock.z - 1) / dimBlock.z);

   rank_3d_filter_kernel<<<dimGrid, dimBlock>>>(d_img, d_norm, width, height, depth, min_intensity);
   getLastCudaError("Error: rankFilter3D() kernel execution FAILED!");
}



__global__ void Rank_Normalize3DKernel(
    float kernel_pixel_num,
    const unsigned short *d_src,
    const float *d_rank,
    float *d_dst, float min_intensity,
    const int width, const int height, const int depth)
{
  const int baseX = blockIdx.x * blockDim.x + threadIdx.x;
  const int baseY = blockIdx.y * blockDim.y + threadIdx.y;
  const int baseZ = blockIdx.z * blockDim.z + threadIdx.z;

  const int idx = (baseZ * height + baseY) * width + baseX;
  const float intensity = (float)d_src[idx];
  d_dst[idx] = (intensity >= min_intensity) ? d_rank[idx] / kernel_pixel_num: 0;
}

__global__ void Copy3DKernel_only(
    const unsigned short *d_src,
    float *d_dst, float min_intensity,
    const int width, const int height, const int depth)
{
  const int baseX = blockIdx.x * blockDim.x + threadIdx.x;
  const int baseY = blockIdx.y * blockDim.y + threadIdx.y;
  const int baseZ = blockIdx.z * blockDim.z + threadIdx.z;

  const int idx = (baseZ * height + baseY) * width + baseX;
  const float intensity = (float)d_src[idx];
  d_dst[idx] = (intensity >= min_intensity) ? intensity : 0;
}

void Rank_Normalize3DFilter(
    unsigned short *d_img, float *d_norm,
    unsigned short *d_erosion_temp1, unsigned short *d_erosion_temp2,
    float *d_erosion_l, float *d_dilation_l,
    float min_intensity,
    const int width, const int height, const int depth,
    const int radius_large_xy, const int radius_large_z)
{
  if (radius_large_xy == 0 || radius_large_z == 0)
  {
    // skip normalize, just copy
    assert(width % (NORM_BLOCKDIM_X) == 0);
    assert(height % (NORM_BLOCKDIM_Y) == 0);
    assert(depth % (NORM_BLOCKDIM_Z) == 0);
    dim3 blocks(width / (NORM_BLOCKDIM_X), height / (NORM_BLOCKDIM_Y), depth / (NORM_BLOCKDIM_Z));
    dim3 threads(NORM_BLOCKDIM_X, NORM_BLOCKDIM_Y, NORM_BLOCKDIM_Z);
    Copy3DKernel_only<<<blocks, threads>>>(d_img,
                                           d_norm, min_intensity,
                                           width, height, depth);
    getLastCudaError("Error: Copy3DKernel() kernel execution FAILED!");
    // checkCudaErrors(cudaDeviceSynchronize());
  }
  else
  {

    float kernel_pixel_num_total = (float)(2*radius_large_xy+1)*(2*radius_large_xy+1)*(2*radius_large_z+1);
    Rank_3DFilter(d_img, d_erosion_l, d_dilation_l,
                              width, height, depth,
                              radius_large_xy, radius_large_z);

    assert(width % (NORM_BLOCKDIM_X) == 0);
    assert(height % (NORM_BLOCKDIM_Y) == 0);
    assert(depth % (NORM_BLOCKDIM_Z) == 0);
    dim3 blocks(width / (NORM_BLOCKDIM_X), height / (NORM_BLOCKDIM_Y), depth / (NORM_BLOCKDIM_Z));
    dim3 threads(NORM_BLOCKDIM_X, NORM_BLOCKDIM_Y, NORM_BLOCKDIM_Z);
    Rank_Normalize3DKernel<<<blocks, threads>>>(kernel_pixel_num_total,d_img,
                                                d_dilation_l,
                                                d_norm, min_intensity,
                                                width, height, depth);

    getLastCudaError("Error: Normalize3DKernel() kernel execution FAILED!");
    // checkCudaErrors(cudaDeviceSynchronize());

        //auto end_time = std::chrono::high_resolution_clock::now();

    //cudaDeviceSynchronize();

    //std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    //double elapsed_time = elapsed.count();
  
    //std::ofstream output_file("/export3/Imaging/Axial/Neurorology/test3/output/a/" + std::to_string(elapsed_time) + "ms.txt", std::ios::app);
    //output_file << elapsed_time << std::endl;
    //output_file.close();
    
   // std::cout << "Block size: " << dimBlock.x << "x" << dimBlock.y << "x" << dimBlock.z << ", Time: " << elapsed_time << " ms" << std::endl;

  }
}
