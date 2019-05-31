/*
  Copyright (c) 2011 Paul Richmond, University of Sheffield , UK; 
  all rights reserved unless otherwise stated.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  In addition to the regulations of the GNU General Public License,
  publications and communications based in parts on this program or on
  parts of this program are required to cite the article 
  "Democratic population decisions result in robust policy-gradient 
  learning: a parametric study with GPU simulations" by Paul Richmond, 
  Lars Buesing, Michele Giugliano and Eleni Vasilaki, PLoS ONE Neuroscience, 
  Under Review.. 

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
  MA 02111-1307 USA
*/

/*
  "This software contains source code provided by NVIDIA Corporation." 
  This CUDA source file defines the host functions used for parallel 
  reduction and is based on the CUDA SDK Parallel reduction example 
  provided with the CUDA Computing SDK. It has been modified to allow a 
  large number of simultaneous (and independent) parallel reductions. i.e.
  multiple reductions.

  The reduction kernel is used for both the spikeTrainRedeuction and outputComponentReduction 
  steps of the simulation.
*/


#ifndef _REDUCTION_H_
#define _REDUCTION_H_

#include "reduction.cuh"

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}



void getReductionBlocksAndThreads(int n, int &blocks, int &threads)
{
	threads = (n < MAX_REDUCTION_THREADS*2) ? nextPow2((n + 1)/ 2) : MAX_REDUCTION_THREADS;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(MAX_REDUCTION_BLOCKS, blocks);
}




template <class T>
void 
reduce(int size, int threads, int blocks, T *d_idata, T *d_odata, int multiple = 1, int total_size = 0)
{
	dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, multiple, 1);
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	//if total size is not default then use mutiple reductions kernel
	if (multiple > 1)
	{
		if (isPow2(size))
		{
			switch (threads)
			{
			case 512:
				reduce6_multiple<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case 256:
				reduce6_multiple<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case 128:
				reduce6_multiple<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case 64:
				reduce6_multiple<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case 32:
				reduce6_multiple<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case 16:
				reduce6_multiple<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case  8:
				reduce6_multiple<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case  4:
				reduce6_multiple<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case  2:
				reduce6_multiple<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case  1:
				reduce6_multiple<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			}
		}
		else
		{
			switch (threads)
			{
			case 512:
				reduce6_multiple<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case 256:
				reduce6_multiple<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case 128:
				reduce6_multiple<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case 64:
				reduce6_multiple<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case 32:
				reduce6_multiple<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case 16:
				reduce6_multiple<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case  8:
				reduce6_multiple<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case  4:
				reduce6_multiple<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case  2:
				reduce6_multiple<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			case  1:
				reduce6_multiple<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, total_size); break;
			}
		}
	}
	//only require a single reduction
	else
	{
		if (isPow2(size))
		{
			switch (threads)
			{
			case 512:
				reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case 256:
				reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case 128:
				reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case 64:
				reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case 32:
				reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case 16:
				reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case  8:
				reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case  4:
				reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case  2:
				reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case  1:
				reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			}
		}
		else
		{
			switch (threads)
			{
			case 512:
				reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case 256:
				reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case 128:
				reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case 64:
				reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case 32:
				reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case 16:
				reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case  8:
				reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case  4:
				reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case  2:
				reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			case  1:
				reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
			}
		}
	}
}


template <class T>
T reduceArray(int n, T* d_idata, T* d_odata)
{
    T gpu_result;

	int nThreads = 0;
	int nBlocks = 0;
	getReductionBlocksAndThreads(n, nBlocks, nThreads);

    //single reduction
	reduce<T>(n, nThreads, nBlocks, d_idata, d_odata);


    // check if kernel execution generated an error
    cutilCheckMsg("Kernel execution failed");

    
    // sum partial block sums on GPU
    int s=nBlocks;
    while(s > 1) 
    {
        int threads = 0, blocks = 0;
        getReductionBlocksAndThreads(s, blocks, threads);
        
		//single reduction
		reduce<T>(s, threads, blocks, d_odata, d_odata);

        s = (s + (threads*2-1)) / (threads*2);

    }
        
	cutilSafeCallNoSync( cudaMemcpy( &gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost) );

    return gpu_result;
}

template <class T>
void reduceMultipleArrays(int n, T* d_idata, T* d_odata, int multiple)
{

	int nThreads = 0;
	int nBlocks = 0;
	getReductionBlocksAndThreads(n, nBlocks, nThreads);

    //mutiple parallel reductions
	reduce<T>(n, nThreads, nBlocks, d_idata, d_odata, multiple, n);

    // check if kernel execution generated an error
    cutilCheckMsg("Kernel execution failed");

    
    // sum partial block sums on GPU
    int s=nBlocks;
    while(s > 1) 
    {
        int threads = 0, blocks = 0;
        getReductionBlocksAndThreads(s, blocks, threads);
        
		//mutiple reductions
		reduce<T>(s, threads, blocks, d_odata, d_odata, multiple, n);
        
        s = (s + (threads*2-1)) / (threads*2);

    }
        
}



template int 
reduceArray<int>(int n, int* d_idata, int* d_odata);

template float 
reduceArray<float>(int n, float* d_idata, float* d_odata);

template double 
reduceArray<double>(int n, double* d_idata, double* d_odata);

template void 
reduceMultipleArrays<int>(int n, int* d_idata, int* d_odata, int multiple);

template void 
reduceMultipleArrays<float>(int n, float* d_idata, float* d_odata, int multiple);

template void 
reduceMultipleArrays<double>(int n, double* d_idata, double* d_odata, int multiple);

#endif //_REDUCTION_H_