/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"


// function for calculating the min; use shared memory and reduce primitive 
__global__ void minmax_shmem_reduce_kernal(float *d_out, const float* d_in, const size_t numRows, const size_t numCols, bool min)
{  
  // the indexes; the "global" location in the whole image
  const int2 pixel_2D_global = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  const int  pixel_1D_global = pixel_2D_global.y * numCols + pixel_2D_global.x;

  // make sure the pixel being processed is indeed within the image
  if (pixel_2D_global.x >= numCols || pixel_2D_global.y >= numRows){
    return;
  }

  // the indexes; the location within the block, i.e, their thread indexes; used to index the sh_image
  const int pixel_1D_block = threadIdx.y * blockDim.x + threadIdx.x;

  // The Shared memory is to store the image block
  extern __shared__ float sh_image[];
  sh_image[pixel_1D_block] = d_in[pixel_1D_global];
  __syncthreads(); // make sure entire block is loaded

  // do reduction in shared mem
  for (int s = blockDim.x * blockDim.y / 2 ; s > 0; s >>= 1){
    if (pixel_1D_block < s){
      if(min){
        if(sh_image[pixel_1D_block + s] < sh_image[pixel_1D_block]){
            sh_image[pixel_1D_block] = sh_image[pixel_1D_block + s];
        }
      }
      else
      {
        if (sh_image[pixel_1D_block + s] > sh_image[pixel_1D_block]){
            sh_image[pixel_1D_block] =  sh_image[pixel_1D_block + s];
        }
      }
    }

    __syncthreads(); // make sure all operations at one stage are done
  }

  // only thread 0 writes the result for this block back to globel mem
  if (pixel_1D_block == 0){
    d_out[ blockIdx.y * gridDim.x  + blockIdx.x] = sh_image[0];
  }

}


// function to calculating calculate the hist gram
__global__ void histo(int * d_bins, const float* d_in, const size_t numRows, const size_t numCols, const size_t numBins, const float *min_logLum, const float *lumRange)
{

  // the indexes; the "global" location in the whole image
  const int2 pixel_2D_global = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  const int  pixel_1D_global = pixel_2D_global.y * numCols + pixel_2D_global.x;

  // make sure the pixel being processed is indeed within the image
  if (pixel_2D_global.x >= numCols || pixel_2D_global.y >= numRows){
    return;
  }

  float lum = d_in[pixel_1D_global];
  int   bin = (lum - *min_logLum ) / *lumRange * numBins;
  atomicAdd(&(d_bins[bin]), 1);
}


// function to do the exclusive scan

  
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum        
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  // 1)
  int blockSizeX = 32;
  int blockSizeY = 32;
  const int MAXTHREADPERBLOCK = 1024;

  const dim3 blockSize(blockSizeX, blockSizeY, 1);

  unsigned int gridX = ( numCols % blockSize.x) == 0 ? ( numCols / blockSize.x ) : ( numCols / blockSize.x  + 1);
  unsigned int gridY = ( numRows % blockSize.y) == 0 ? ( numRows / blockSize.y ) : ( numRows / blockSize.y  + 1);
  const dim3 gridSize( gridX, gridY, 1); 

  // declare intermediate GPU memory pointers
  float *d_intermediate, *d_out;

  // allocate memory for them
  checkCudaErrors( cudaMalloc( (void **) &d_intermediate,  sizeof(float) *gridX * gridY ) );
  checkCudaErrors( cudaMalloc( (void **) &d_out,  sizeof(float)));

  // min calculation
  minmax_shmem_reduce_kernal<<<gridSize, blockSize, sizeof(float) * blockSizeX * blockSizeY >>> (d_intermediate, d_logLuminance, numRows, numCols, true);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  //minmax_shmem_reduce_kernal<<<1, gridX * gridY,  sizeof(float) * gridX * gridY>>> (d_out, d_intermediate, gridY, gridX, true);
 
  min_logLum = *d_out;

  //max calculation
  minmax_shmem_reduce_kernal<<<gridSize, blockSize, sizeof(float) * blockSizeX * blockSizeY >>> (d_intermediate, d_logLuminance, numRows, numCols, false);

  //minmax_shmem_reduce_kernal<<<1, gridX * gridY,  sizeof(float) * gridX * gridY>>> (d_out, d_intermediate, gridY, gridX, false); 

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  max_logLum = *d_out;

  // 2)
  const float lumRange = max_logLum - min_logLum;

  // 3)
  // declare GPU memory
  int* d_bins;

  //allocate memory for it;
  checkCudaErrors( cudaMalloc( (void **) & d_bins, sizeof(int) * numBins ));
  
  // initialize to 0 
  for(int i = 0 ; i < numBins; ++i){
    d_bins[i] = 0;
  }

  histo<<<gridSize, blockSize>>>(d_bins, d_logLuminance, numRows, numCols, numBins, &min_logLum, &lumRange);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  // 4)
  int acc = 0;
  for(int i = 0 ; i < numBins; ++i){
    d_cdf[i] = acc;
    acc += d_bins[i];
  }

  // Free memory
  checkCudaErrors(cudaFree(d_intermediate));
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(d_bins));

}