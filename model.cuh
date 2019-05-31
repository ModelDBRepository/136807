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
This CUDA source file contains all the GPU device kernels which correspond 
with those in Figure ** of the accompanying paper. The kernels which use 
matrix multiply and matrix transpose are based on the CUDA SDK examples provided 
with the CUDA Computing SDK. Where applicable they have been modified to allow
a large number of simultaneous (and independent) matrix multiplications/transpose 
operations. A number of macro definitions are also specified which either are 
provided in order to make reading the source easier or optimise a particular operation.
*/

#ifndef _BANDIT_CUH
#define _BANDIT_CUH

#include <random.h>
#include <random.cuh>
#include <parameters.cuh>


//CUDA specific paramters
#define THREADS_PER_BLOCK 256	//(This may need to be reduced to 128 on older non Fermi cards)
#define SM_CHUNK_SIZE 32		//M & N must be multiples of this to perform matrix operations

//CUDA defines (to make source more legible)
#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z
#define SDATA( index)      cutilBankChecker(sdata, index)
#define M_BLOCK_SIZE MIN(THREADS_PER_BLOCK, M)					//min of M or THREADS_PER_BLOCK
#define N_BLOCK_SIZE MIN(THREADS_PER_BLOCK, N)					//min of M or THREADS_PER_BLOCK
#define Tr_Analysis_MAX MAX(analysis_trials, gradiant_analsyis_trials)
#define Tr_Learn_MAX MAX(learn_trials_dyn, learn_trials_no_dyn)
#define Tr_MAX MAX(Tr_Analysis_MAX, Tr_Learn_MAX)			

//Bit shifting operations to provide efficient mod and divide for power of 2 numbers (hackety hack!)
#define pow2mod(x, y) (x & (y-1))	//x>0 && y must be pow 2
#define epsilon 0.0001f
#define intdivf(x, y) (int)(((float)x/(float)y)+epsilon)



/** logi
 *	logistic function
 */
__device__ float logi(float x)
{
	return 1.0f/(1.0f+expf(-x));
}


__global__ void resetMembranePotential(float* d_u, float* d_u_out)
{
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int d_u_offset = (T*N*blockIdx.y);
	unsigned int configuration_offset = pow2mod(blockIdx.y, ind_configs);

	d_u[index + d_u_offset]		= urest[configuration_offset];
	d_u_out[index + d_u_offset] = urest[configuration_offset];
}


/** poissonNeuronSimulation
 *	
 */
__global__ void poissonNeuronSimulation(int n, int Tr, float* d_X, float* d_Input, float* d_epsp, float* d_In_Deg, rand48seeds* seeds, magic_numbers mn)
{
	//calculate index (max of M)
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int d_X_offset = (T*M*blockIdx.y); //offset for each trial
	unsigned int d_Input_offset = (T*M*blockIdx.y); //offset for each trial
	unsigned int d_In_Deg_offset = (Tr*blockIdx.y); //offset for each trial
	unsigned int configuration_offset = pow2mod(blockIdx.y, ind_configs);


	//calculate distance between target and prefered angle
	float d = fabs(index*2.0f*PI/M - d_In_Deg[n+d_In_Deg_offset]);	//offset by Tr for each ind trial

	//calculate activation value
	//split up original calculation into conditional format to increase thread coherance
	float Xf;
	if (d<PI)
		Xf = logi((PI/M-d)/beta[configuration_offset]); 
	else	//(d>=PI)
		Xf = logi((d-PI*(2*M-1)/M)/beta[configuration_offset]);

	//previous input
	float epsp = 0;

	//output activations applying some random noise then calculate spike ouputs
	for (int t=0; t<T; t++)
	{
		//activations
		float activation = Xf*xsc[configuration_offset]*dt;
		d_X[index+(t*M)+d_X_offset] = activation;

		//spikes
		float random = rand48(seeds, mn, M);
		float input = (random < activation);
		d_Input[index+(t*M)+d_Input_offset] = input;

		//epsp
		epsp = input + epsp*lambda[configuration_offset];
		d_epsp[index+(t*M)+d_Input_offset] = epsp;
	}
}

/** placeCellSpikePropagation
 *
 */
__global__ void placeCellSpikePropagation(float* d_W, float* d_X, float* d_input_pot)
{
	__shared__ float buff[M_BLOCK_SIZE];
    unsigned int rowIdx = __mul24(M,blockIdx.x);
	unsigned int d_W_offset = (M*N*blockIdx.y); //offset for each trial
	unsigned int d_X_offset = (M*T*blockIdx.y); //offset for each trial
	unsigned int d_input_pot_offset = (N*blockIdx.y); //offset for each trial

    // save sums into shared memory
    buff[threadIdx.x] = 0.0f;
    for(int idx = threadIdx.x; idx < M; idx += blockDim.x) {
        float Aval   = d_W[rowIdx+idx+d_W_offset];
        float xval   = d_X[idx+d_X_offset];
        buff[threadIdx.x] += Aval * xval;
    }
	
    __syncthreads();

    // use a parallel reduction
    // partial sum in shared memory.
    if (threadIdx.x < 32) {
        #pragma unroll 
        for(int i=32; i<M_BLOCK_SIZE; i+=32) buff[threadIdx.x] += buff[threadIdx.x+i]; 
    }
    if (threadIdx.x < 16) { buff[threadIdx.x] += buff[threadIdx.x+16]; }
    if (threadIdx.x < 8)  { buff[threadIdx.x] += buff[threadIdx.x+8]; }
    if (threadIdx.x < 4)  { buff[threadIdx.x] += buff[threadIdx.x+4]; }
    if (threadIdx.x < 2)  { buff[threadIdx.x] += buff[threadIdx.x+2]; }

    // sum the last two values and write to global memory
    if (threadIdx.x == 0)  { 
        d_input_pot[blockIdx.x+d_input_pot_offset] = buff[0] + buff[1];
    }
}

/** actionCellLateralSpikePropagation
 * 
 */
__global__ void actionCellLateralSpikePropagation(float* d_Wr, float* d_Y, float* d_lateral_pot)
{
	__shared__ float buff[N_BLOCK_SIZE];

    unsigned int rowIdx = __mul24(N,blockIdx.x);
	unsigned int d_Y_offset = (N*T*blockIdx.y); //offset for each trial
	unsigned int d_lateral_pot_offset = (N*blockIdx.y); //offset for each trial

	unsigned int configuration_offset = pow2mod(blockIdx.y, ind_configs);
	unsigned int d_Wr_offset = configuration_offset*N*N;


    // save sums into shared memory
    buff[threadIdx.x] = 0.f;
    for(int idx = threadIdx.x; idx < N; idx += blockDim.x) {
		float Aval   = d_Wr[rowIdx+idx+d_Wr_offset];
        float xval   = d_Y[idx+d_Y_offset];
        buff[threadIdx.x] += Aval * xval;
    }
	
    __syncthreads();

    // use a parallel reduction
    if (threadIdx.x < 32) {
        #pragma unroll 
        for(int i=32; i<N_BLOCK_SIZE; i+=32) buff[threadIdx.x] += buff[threadIdx.x+i]; 
    }
    if (threadIdx.x < 16) { buff[threadIdx.x] += buff[threadIdx.x+16]; }
    if (threadIdx.x < 8)  { buff[threadIdx.x] += buff[threadIdx.x+8]; }
    if (threadIdx.x < 4)  { buff[threadIdx.x] += buff[threadIdx.x+4]; }
    if (threadIdx.x < 2)  { buff[threadIdx.x] += buff[threadIdx.x+2]; }

    // sum the last two values and write to global memory
    if (threadIdx.x == 0)  { 
        d_lateral_pot[blockIdx.x+d_lateral_pot_offset] = buff[0] + buff[1];
    }
}

/**
 * integrateAndFireNeuronSimulation
 */
template <DYNAMICS dynamics>
__global__ void integrateAndFireNeuronSimulation(int t, float* d_u, float* d_u_out, float* d_input_pot, float* d_lateral_pot, float* d_YProb, float* d_Y, rand48seeds* seeds, magic_numbers mn)
{
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + tx;	//global index [0:N] 

	unsigned int NT_offset = (N*T*blockIdx.y); //offset for each trial
	unsigned int N_offset = (N*blockIdx.y); //offset for each trial

	unsigned int configuration_offset = pow2mod(blockIdx.y, ind_configs);

	//calculate membrane potential
	float u;
	if (dynamics == DYN_SYS)
	{
		u = urest[configuration_offset] + (d_u[index+((t-1)*N)+NT_offset] - urest[configuration_offset])*lambda[configuration_offset] 
										  + d_input_pot[index+N_offset] 
										  + d_lateral_pot[index+N_offset];
	}else	//NO_DYN_SYS 
	{
		u = urest[configuration_offset] + (d_u[index+((t-1)*N)+NT_offset] - urest[configuration_offset])*lambda[configuration_offset] 
										  + d_input_pot[index+N_offset];
	}

	float rho = rho0[configuration_offset]*expf((u-threshold[configuration_offset])/du[configuration_offset]);


	//calculate the actual probability of firing
	float YProb = 1.0f-exp(-dt*rho);
	d_YProb[index+(t*N)+NT_offset] = YProb;

	//calculate action cell spikes probability (include some noise)
	float rand = rand48(seeds, mn, N);
	float Y = rand < YProb;
	d_Y[index+(t*N)+NT_offset] = Y;

	//apply refactory reset to mebrane potential of firing neurons
	if (REFACTORY){
		if (Y > 0.5f)
			u = (urest[configuration_offset]-ref[configuration_offset]);
	}
	//save u
	d_u_out[index+(t*N)+NT_offset] = u;
}


/** Generic transpose
 * Coalseced Matrix transpose (no conflicts) from CUDA SDK examples 
 */
template <int width, int height>
__global__ void transpose(float *d_A, float *d_A_Trans, int grid_width)
{
	__shared__ float block[SM_CHUNK_SIZE][SM_CHUNK_SIZE+1];

	unsigned int d_A_offset = (width*height*blockIdx.y); //offset for each trial
	unsigned int d_A_Trans_offset = (width*height*blockIdx.y); //offset for each trial
	
	int bx = pow2mod(blockIdx.x, grid_width);
	int by = intdivf(blockIdx.x, grid_width);

	// read the matrix tile into shared memory
	unsigned int xIndex = bx * SM_CHUNK_SIZE + tx;
	unsigned int yIndex = by * SM_CHUNK_SIZE + ty;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[ty][tx] = d_A[index_in+d_A_offset];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = by * SM_CHUNK_SIZE + tx;
	yIndex = bx * SM_CHUNK_SIZE + ty;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		d_A_Trans[index_out+d_A_Trans_offset] = block[tx][ty];
	}
}


/** calculatePopulationVector
 *
 */
__global__ void calculatePopulationVector(float* d_Y_sum, float* d_out_x, float* d_out_y)
{
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + tx;

	unsigned int d_Y_sum_offset = (N*T*blockIdx.y); //offset for each trial
	unsigned int d_out_x_offset = (N*blockIdx.y); //offset for each trial
	unsigned int d_out_y_offset = (N*blockIdx.y); //offset for each trial

	//d_Y_sum has N average values every T elements (the rest is junk) 
	float Y_sum = d_Y_sum[index*T+d_Y_sum_offset];
	float Y_mean = Y_sum/T;

	d_out_x[index+d_out_x_offset] = sinf(2.0f*PI*(float)index/N) * Y_mean;
	d_out_y[index+d_out_y_offset] = cosf(2.0f*PI*(float)index/N) * Y_mean;
}

/** calculateGradiant
 * calculate gradiant (plain old matrix multiply avoiding shared memory bank conflicts)
 */
__global__ void calculateGradiant(int grid_width, float* d_Yt, float* d_YProbt, float* d_X, float* D_Grad)
{

	unsigned int d_Yt_offset = (N*T*blockIdx.y); //offset for each trial
	unsigned int d_YProbt_offset = (N*T*blockIdx.y); //offset for each trial
	unsigned int d_X_offset = (M*T*blockIdx.y); //offset for each trial
	unsigned int d_Grad_offset = (N*M*blockIdx.y); //offset for each trial


	int bx = pow2mod(blockIdx.x, grid_width);
	int by = intdivf(blockIdx.x, grid_width);
	//chunk step sizes
    int aBegin = T * SM_CHUNK_SIZE * by;
	int aEnd   = aBegin + T - 1;
    int aStep  = SM_CHUNK_SIZE;
    int bBegin = SM_CHUNK_SIZE * bx;
    int bStep  = SM_CHUNK_SIZE * M;

    float grad = 0;

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a=aBegin, b=bBegin; a<=aEnd; a+=aStep, b+=bStep) {

		//add shared memory to avoid conflicts
        __shared__ float As[SM_CHUNK_SIZE][SM_CHUNK_SIZE+1];
        __shared__ float Bs[SM_CHUNK_SIZE][SM_CHUNK_SIZE+1];

        As[ty][tx] = d_Yt[a + T * ty + tx + d_Yt_offset] - d_YProbt[a + T * ty + tx + d_YProbt_offset];
        Bs[ty][tx] = d_X[b + M * ty + tx + d_X_offset];
        __syncthreads();

        for (int k = 0; k < SM_CHUNK_SIZE; ++k)
            grad += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    // Write the block sub-matrix to device memory each thread writes one element
    int c = M * SM_CHUNK_SIZE * by + SM_CHUNK_SIZE * bx;
	D_Grad[c + M * ty + tx + d_Grad_offset] = grad;
}

/** updateLearningWeghts
 *
 */
__global__ void updateLearningWeights(float* d_W, float* reward, float* d_Grad, float* d_W_out)
{
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + tx;

	unsigned int d_W_offset = (N*M*blockIdx.y); //offset for each trial
	unsigned int d_Grad_offset = (N*M*blockIdx.y); //offset for each trial
	unsigned int configuration_offset = pow2mod(blockIdx.y, ind_configs);



	float W = d_W[index + d_W_offset];
	float gradiant = d_Grad[index + d_Grad_offset];

	//learning rule
	float W_out = W + eta[configuration_offset]*(reward[blockIdx.y]-baseline[configuration_offset])* gradiant;

	//clamp W_out within range of -1 and 1
	W_out = (W_out<-1)? -1 : W_out;
	W_out = (W_out >1)? 1 : W_out;

	d_W_out[index + d_W_offset] = W_out; 
}

__global__ void applyNoise(float* d_W, float* d_W_out, rand48seeds* seeds, magic_numbers mn)
{
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + tx;

	unsigned int d_W_offset = (N*M*blockIdx.y); //offset for each trial
	unsigned int configuration_offset = pow2mod(blockIdx.y, ind_configs);

	
	//apply noise accross population
	for (int i=0; i<N; i++)
	{
		float W = d_W[index + (i*M) + d_W_offset];
		
		float noise = rand48(seeds, mn, M);
		
		float W_out = W + (noise-0.25f)*w0[configuration_offset];

		//clamp W_out within range of -1 and 1
		W_out = (W_out<-1)? -1 : W_out;
		W_out = (W_out >1)? 1 : W_out;

		//write W out
		d_W_out[index + (i*M) + d_W_offset] = W_out;
	}
}

__global__ void updateGradiantAnalysis(int* d_decision_offset, int* d_target_offset, float* d_Grad, float* reward, float* d_sumGrad, float* d_sumGrad_out, float* d_sumDeltaW, float* d_sumDeltaW_out)
{
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + tx;
	unsigned int d_Grad_offset = (N*M*blockIdx.y); //offset for each trial
	unsigned int configuration_offset = pow2mod(blockIdx.y, ind_configs);

	//index position in NxM grid
	unsigned int m_index = pow2mod(index, M);
	unsigned int n_index = intdivf(index, M);

	//descision offset
	unsigned int decision_offset = d_decision_offset[blockIdx.y];
	unsigned int descision_offset_n_index = pow2mod((n_index+decision_offset), N);
	unsigned int decision_offset_index = (descision_offset_n_index*M)+m_index;

	//target offset
	unsigned int target_offset = d_target_offset[blockIdx.y];
	unsigned int target_offset_n_index = pow2mod((n_index+target_offset), N);
	unsigned int target_offset_index = (target_offset_n_index*M)+m_index;

	//sum the gradiant (xi-P(xi))*xj and offset by decision offset
	float decision_gradiant = d_Grad[decision_offset_index + d_Grad_offset];
	d_sumGrad_out[index + d_Grad_offset] = d_sumGrad[index + d_Grad_offset] + decision_gradiant;

	//sum the change in weight (xi-P(xi))*xj and offset by target offset
	float target_gradiant = d_Grad[target_offset_index + d_Grad_offset];

	d_sumDeltaW_out[index + d_Grad_offset] = d_sumDeltaW[index + d_Grad_offset] + eta[configuration_offset]*(reward[blockIdx.y]-baseline[configuration_offset])*target_gradiant;
	
}

#endif // #ifndef _BANDIT_CUH 



