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


#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "utils.h"


//
int get_max_size(int n, int d) {
    return (int)ceil( (float)n/(float)d ) ;
}


// function for calculating the min; use shared memory and reduce primitive 
__global__ void minmax_shmem_reduce_kernal(float *d_out, const float* d_in, const int NUMTOLPIXEL, bool min)
{  
  // the indexes; the "global" location in the whole image
  const int  pixel_1D_global = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure the pixel being processed is indeed within the image
  if (pixel_1D_global >= NUMTOLPIXEL){
    return;
  }

  // the indexes; the location within the block, i.e, their thread indexes; used to index the sh_image
  const int pixel_1D_block = threadIdx.x;

  // The Shared memory is to store the image block
  extern __shared__ float sh_image[];
  sh_image[pixel_1D_block] = d_in[pixel_1D_global];
  __syncthreads(); // make sure entire block is loaded

  // do reduction in shared mem
  for (int s = blockDim.x / 2 ; s > 0; s >>= 1){
    if (pixel_1D_block < s){
      if(min){
		sh_image[pixel_1D_block] = fmin(sh_image[pixel_1D_block], sh_image[pixel_1D_block + s]);
      }
      else
      {
		sh_image[pixel_1D_block] = fmax(sh_image[pixel_1D_block], sh_image[pixel_1D_block + s]);
      }
    }
    __syncthreads(); // make sure all operations at one stage are done
  }

  // only thread 0 writes the result for this block back to globel mem
  if (pixel_1D_block == 0){
    d_out[blockIdx.x] = sh_image[0];
  }

}


// function to calculating calculate the hist gram
__global__ void histo_kernel(int * d_bins, const float* d_in, const size_t NUMTOLPIXEL, const size_t numBins, const float min_logLum, const float lumRange)
{

  // the indexes; the "global" location in the whole image
  const int  pixel_1D_global = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure the pixel being processed is indeed within the image
  if (pixel_1D_global >= NUMTOLPIXEL){
    return;
  }

  float lum = d_in[pixel_1D_global];
  int   bin = (lum - min_logLum ) / lumRange * numBins;
  atomicAdd(&(d_bins[bin]), 1);
}

// function to calculating the cdf
__global__ void scancdf_kernel(int *d_bins, int size)
{
    int mid = threadIdx.x + blockDim.x * blockIdx.x;
    if(mid >= size)
        return;
    
    for(int s = 1; s <= size; s *= 2) {
          int spot = mid - s; 
         
          unsigned int val = 0;
          if(spot >= 0)
              val = d_bins[spot];
          __syncthreads();
          if(spot >= 0)
              d_bins[mid] += val;
          __syncthreads();

    }

}



// function to calculate min or max using reduce
float reduce_minmax(const float* const d_logLuminance, const size_t NUMTOLPIXEL, bool minFlag)
{

	const int BLOCKSIZE =   1024;
  
	size_t curr_size = NUMTOLPIXEL;
    printf("Curr_size = %d\n", curr_size);

	// declare intermediate GPU memory pointers
	float *d_out, *d_in;

	// allocate memory for them
	checkCudaErrors(cudaMalloc( (void **) &d_in, sizeof(float) * NUMTOLPIXEL));    
	checkCudaErrors(cudaMemcpy(d_in, d_logLuminance, sizeof(float) * NUMTOLPIXEL, cudaMemcpyDeviceToDevice));
 
	dim3 thread_dim(BLOCKSIZE);


	while(curr_size > 1 ){
		checkCudaErrors( cudaMalloc( (void **) &d_out,  sizeof(float) * get_max_size(curr_size, BLOCKSIZE) ) );

		
		dim3 block_dim(get_max_size(curr_size, BLOCKSIZE));

		minmax_shmem_reduce_kernal<<<block_dim, thread_dim, sizeof(float) * BLOCKSIZE >>> (d_out, d_in, curr_size, minFlag);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaFree( d_in) );
		d_in = d_out;

		curr_size = get_max_size(curr_size, BLOCKSIZE);

		printf("Curr_size = %d\n", curr_size);

	}

	float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
	
    cudaFree(d_out);
	
	printf("%f\n", h_out);
	
    return h_out;
	
}




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

  // ZL: Use 1D block and thread to simply the indexing through out the whole file
	const size_t NUMTOLPIXEL  = numRows * numCols;
	const int MAXTHREADPERBLOCK = 1024;	

  // 1) *********************************************  1)
	min_logLum = reduce_minmax(d_logLuminance, NUMTOLPIXEL,true);
	max_logLum = reduce_minmax(d_logLuminance, NUMTOLPIXEL,false);

  // ************************************************  2)
	const float lumRange = max_logLum - min_logLum;

  // ************************************************  3)
  // declare GPU memory
	int* d_bins;
  
  //allocate memory for it;
	checkCudaErrors( cudaMalloc( (void **) & d_bins, sizeof(int) * numBins ));
  
  // initialize to 0 
    checkCudaErrors(cudaMemset(d_bins, 0, sizeof(int) * numBins));  

  // Call histo kernel
	const int blockSize =  MAXTHREADPERBLOCK;
	int gridSize  = get_max_size( NUMTOLPIXEL, MAXTHREADPERBLOCK);
	dim3 thread_dim(blockSize);
	dim3 block_dim( gridSize );
	histo_kernel<<<block_dim, thread_dim>>>(d_bins, d_logLuminance, NUMTOLPIXEL, numBins, min_logLum, lumRange);
	
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  // Scan to get CDF
	gridSize = get_max_size(numBins, blockSize);
	dim3 block_dim_cdf( gridSize );
	scancdf_kernel<<<block_dim_cdf, thread_dim>>>(d_bins,  numBins);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	cudaMemcpy(d_cdf, d_bins, sizeof(int) * numBins, cudaMemcpyDeviceToDevice);
  // Free memory
  checkCudaErrors(cudaFree(d_bins));

}