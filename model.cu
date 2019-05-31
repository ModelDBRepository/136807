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


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <time.h>
#include <float.h>
#include <cutil_inline.h>

// includes, project
#include "reduction.cuh"
#include "model.h"
#include "model.cuh"
#include "output.cuh"

////////////////////////////////////////////////////////////////////////////////
//Macro definitions
#define MAX(x,y) ((x > y) ? x : y)
#define MAX_NEURONS MAX(M, N)
#define MAX_RAND MAX_NEURONS //MAX(MAX_NEURONS, SM_CHUNK_SIZE*SM_CHUNK_SIZE)	//factor of T required for no dyn, N*M for weight update noise



bool printWeightPlot;

////////////////////////////////////////////////////////////////////////////////
// Thread/Grid blocks
dim3 m_threads;	
dim3 m_grid;

dim3 n_threads;	
dim3 n_grid;

dim3 nN_threads;
dim3 nN_grid;

dim3 mN_threads;
dim3 mN_grid;

dim3 NT_matrix_threads;	
dim3 NT_matrix_grid;	
int NT_grid_width;

dim3 NT_threads;
dim3 NT_grid;

dim3 MN_matrix_threads;
dim3 MN_matrix_grid;
int MN_grid_width;

dim3 MN_threads;
dim3 MN_grid;

//////////////////////////////////////////////////////
//Persitant global variables
float* d_Wr;
float* d_W;
float* d_W_out;
float* d_Wt;
//Step variables
float* h_In_Deg;	//host data
float* d_In_Deg;
float* d_rew;
int* d_decision_offset;
int* d_target_offset;
//Trial variables
float* d_u;
float* d_u_out;
float* d_Y;
float* d_Yt;
float* d_Y_sum;
float* d_YProbt;
float* d_input_pot;
float* d_lateral_pot;
float* d_YProb;
float* d_X;	
float* d_Input;	
float* d_epsp;
float* d_Grad;
float* d_out_x;
float* d_out_x_sum;
float* d_out_y;
float* d_out_y_sum;
rand48seeds* d_randSeeds;
magic_numbers mn;

//data used for producing gradiant graphs
float* h_W;
float* d_sumGrad;
float* d_sumDeltaW;
float* d_sumGrad_out;
float* d_sumDeltaW_out;
float* h_sumGrad;
float* h_sumDeltaW;



/**	Reward function  */
float reward(float x, int config)
{
	float reward = 0;

	//guassian reward
	if (reward_func == GAUSSIAN)
	{
		if (x<PI)
			reward = exp(-pow(x, 2.0f)/(2.0f*pow(sigma_R[config], 2.0f)));
		else
			reward = exp(-pow(2.0f*PI-x, 2.0f)/(2.0f*pow(sigma_R[config], 2.0f))); 

	}
	//box reward
	else if (reward_func == BOX)
	{
		if (x<PI)
			reward = (float)(x<sigma_R_box[config])*(x>-sigma_R_box[config]);
		else
			reward = (float)((2.0f*PI-x)<sigma_R_box[config]) * ((2.0f*PI-x)>-sigma_R_box[config]);
	}

	return reward;
}


float errorFunc(float x)
{

	float err;
	
	if (x<PI)
		err = x;
	else
		err = (2.0f*PI-x);

	return err;
}

/* Mod function using sign from divisor (as with python)*/
float py_modf(float n, float a) {
    float r = fmodf(n, a);
	float sign = (a > 0) ? 1.0f : -1.0f;
    if (r * sign < 0) r += a;
    return r;
}


void copyParametersToDevice()
{
	cutilSafeCall( cudaMemcpyToSymbol(beta, &h_beta, sizeof(float)*ind_configs));
	cutilSafeCall( cudaMemcpyToSymbol(xsc, &h_xsc, sizeof(float)*ind_configs));
	cutilSafeCall( cudaMemcpyToSymbol(u0, &h_u0, sizeof(float)*ind_configs));
	cutilSafeCall( cudaMemcpyToSymbol(w0, &h_w0, sizeof(float)*ind_configs));
	cutilSafeCall( cudaMemcpyToSymbol(eta, &h_eta, sizeof(float)*ind_configs));
	cutilSafeCall( cudaMemcpyToSymbol(baseline, &h_baseline, sizeof(float)*ind_configs));

	cutilSafeCall( cudaMemcpyToSymbol(tau, &h_tau, sizeof(float)*ind_configs));
	cutilSafeCall( cudaMemcpyToSymbol(lambda, &h_lambda, sizeof(float)*ind_configs));
	cutilSafeCall( cudaMemcpyToSymbol(urest, &h_urest, sizeof(float)*ind_configs));
	cutilSafeCall( cudaMemcpyToSymbol(threshold, &h_threshold, sizeof(float)*ind_configs));
	cutilSafeCall( cudaMemcpyToSymbol(du, &h_du, sizeof(float)*ind_configs));
	cutilSafeCall( cudaMemcpyToSymbol(rho0, &h_rho0, sizeof(float)*ind_configs));
	cutilSafeCall( cudaMemcpyToSymbol(ref, &h_ref, sizeof(float)*ind_configs));
}


/** initLearnStepData
 * Initialises all device data used during the learn_step function.
 * Initialises Weight matrices W (+W_out) and Wr on the device. Wr has static laternal connection weights. W is set to zero. 
 * A seperate set of data is created for each indipendant trial excluding Wr which is the same throughout all.
 * A set of host data is allocated (h_In_Deg) which is used in the learn step to create random goals.
 */
void initLearnStepData()
{
	//allocate W matrix on host to save weights later on
	if (printWeightPlot)
	{
		h_W = (float*) malloc(N*M*sizeof(float)*ind_trials);
	}

	//allocate weight data
	cudaMalloc( (void**) &d_Wr, N*N*sizeof(float)*ind_configs );
	cudaMalloc( (void**) &d_W, M*N*sizeof(float)*ind_trials);
	cudaMalloc( (void**) &d_W_out, M*N*sizeof(float) *ind_trials);
	cudaMalloc( (void**) &d_Wt, M*N*sizeof(float) *ind_trials);

	//init lateral connection weights and copy to device
	float* h_Wr = (float*) malloc(N*N*sizeof(float)*ind_configs);
	for(int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
		{
			float dist = abs(j-i*1.0f); 
			dist = dist*(dist<(N/2)) + (N-dist)*(dist>=(N/2));

			//Wr varies for each set of configuration parameters
			for(int k=0; k<ind_configs; k++)
			{
				h_Wr[j+(i*N) + (k*N*N)] =	(expf(-powf(dist,2.0f)/(2.0f*powf(sig_p[k],2.0f)))*w_E[k]  - 0.9f)*rsc[k];		//j+(i*N) possibly should be i+(j*N)
			}
		}
	}
	cutilSafeCall( cudaMemcpy(d_Wr, h_Wr, N*N*sizeof(float)*ind_configs, cudaMemcpyHostToDevice) );
	free(h_Wr);

	//Set action cell weights to zero
	cutilSafeCall( cudaMemset(d_W, 0,M*N*sizeof(float)*ind_trials));
	cutilSafeCall( cudaMemset(d_W_out, 0,M*N*sizeof(float)*ind_trials));
	cutilSafeCall( cudaMemset(d_Wt, 0,M*N*sizeof(float)*ind_trials));

	//allocate global learnstep variables
	cudaMalloc( (void**) &d_In_Deg, Tr_MAX*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_rew, sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_decision_offset, sizeof(int)*ind_trials );
	cudaMalloc( (void**) &d_target_offset, sizeof(int)*ind_trials );
	

	//allocate trial specific data
	cudaMalloc( (void**) &d_u, T*N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_u_out, T*N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_Y, T*N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_Yt, T*N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_Y_sum, T*N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_YProb, T*N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_YProbt, T*N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_input_pot, T*N*sizeof(float)*ind_trials );	//factor of T only required for no dyn system
	cudaMalloc( (void**) &d_lateral_pot, N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_X, T*M*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_Input, T*M*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_epsp, T*M*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_Grad, M*N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_out_x, N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_out_x_sum, N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_out_y, N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_out_y_sum, N*sizeof(float)*ind_trials );

	//init host data used to create a set of random goals 
	h_In_Deg = (float*) malloc(Tr_MAX*sizeof(float)*ind_trials);

	//init randomn number stream data using rand48 algorithm
	rand48seeds* h_randSeeds;
	h_randSeeds = (rand48seeds*) malloc(MAX_RAND*sizeof(rand48seeds)*ind_trials);
	cudaMalloc( (void**) &d_randSeeds, MAX_RAND*sizeof(rand48seeds)*ind_trials);
	initCUDARand48(MAX_RAND*ind_trials, h_randSeeds, d_randSeeds, mn);
	free(h_randSeeds);
}

void initGraphAnalysis()
{
	//malloc data on host
	h_sumGrad = (float*)malloc(M*N*sizeof(float)*ind_trials);
	h_sumDeltaW = (float*)malloc(M*N*sizeof(float)*ind_trials);
	
	//malloc dat on device
	cudaMalloc( (void**) &d_sumGrad, M*N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_sumDeltaW, M*N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_sumGrad_out, M*N*sizeof(float)*ind_trials );
	cudaMalloc( (void**) &d_sumDeltaW_out, M*N*sizeof(float)*ind_trials );
}

void resetGraphAnalysis()
{
	//block set device data to 0
	cutilSafeCall( cudaMemset(d_sumGrad, 0,M*N*sizeof(float)*ind_trials));
	cutilSafeCall( cudaMemset(d_sumDeltaW, 0,M*N*sizeof(float)*ind_trials));
	cutilSafeCall( cudaMemset(d_sumGrad_out, 0,M*N*sizeof(float)*ind_trials));
	cutilSafeCall( cudaMemset(d_sumDeltaW_out, 0,M*N*sizeof(float)*ind_trials));
}

void cleanupGraphAnalysis()
{
	//free host memory
	free(h_sumGrad);
	free(h_sumDeltaW);

	//free device memory
	cudaFree(d_sumGrad);
	cudaFree(d_sumDeltaW);
	cudaFree(d_sumGrad_out);
	cudaFree(d_sumDeltaW_out);
}


/** cleanupLearnStep
 *	Frees all data on host and device which is used in the learn step.
 */
void cleanupLearnStep()
{
	//dealloc learn step data
	free(h_In_Deg); 
	cudaFree(d_In_Deg);
	cudaFree(d_rew);
	cudaFree(d_decision_offset);
	cudaFree(d_target_offset);
	

	//dealloc trial data
	cudaFree (d_randSeeds);
	cudaFree(d_u);
	cudaFree(d_u_out);
	cudaFree(d_Y);
	cudaFree(d_Yt);
	cudaFree(d_Y_sum);
	cudaFree(d_YProb);
	cudaFree(d_YProbt);
	cudaFree(d_input_pot);
	cudaFree(d_lateral_pot);
	cudaFree(d_X);
	cudaFree(d_Input);
	cudaFree(d_epsp);
	cudaFree(d_Grad);
	cudaFree(d_out_x);
	cudaFree(d_out_x_sum);
	cudaFree(d_out_y);
	cudaFree(d_out_y_sum);

	//dealloc weight data
	if (printWeightPlot)
		free(h_W);
	cudaFree( d_W);
	cudaFree( d_W_out);
	cudaFree( d_Wr);
	cudaFree( d_Wt);

}

/** calculateGridBlockSizes
 * All grid block sizes are pre-calculated before the learn step
 */
void calculateGridBlockSizes()
{
	//M total threads with a max block size of M
	m_threads = dim3(M_BLOCK_SIZE, 1, 1);	
	m_grid = dim3(M/M_BLOCK_SIZE, ind_trials, 1);

	//N total threads with a max block size of N
	n_threads = dim3(N_BLOCK_SIZE, 1, 1);	
	n_grid = dim3(N/N_BLOCK_SIZE, ind_trials, 1);

	//n block size by N total threads
	nN_threads = dim3(N_BLOCK_SIZE, 1 , 1);
	nN_grid = dim3(N, ind_trials, 1);

	//m block size by N total threads
	mN_threads = dim3(M_BLOCK_SIZE,1 , 1);
	mN_grid = dim3(N, ind_trials, 1);

	//NxT total threads with SM_CHUNK_SIZE^2 threads per block
	NT_matrix_threads = dim3(SM_CHUNK_SIZE, SM_CHUNK_SIZE, 1);	
	NT_matrix_grid = dim3(N*T/(SM_CHUNK_SIZE*SM_CHUNK_SIZE), ind_trials, 1);
	NT_grid_width = N/SM_CHUNK_SIZE;

	//N*T total threads
	NT_threads = dim3(SM_CHUNK_SIZE *SM_CHUNK_SIZE, 1, 1);	
	NT_grid = dim3(N*T/(SM_CHUNK_SIZE*SM_CHUNK_SIZE), ind_trials, 1);

	//NxM total threads with SM_CHUNK_SIZE^2 threads per 2D block
	//both x and y of grid are held within the x dimenion of the grid (requires mod and divide in kernel)
	MN_matrix_threads = dim3(SM_CHUNK_SIZE, SM_CHUNK_SIZE, 1);
	MN_matrix_grid = dim3(M*N/(SM_CHUNK_SIZE*SM_CHUNK_SIZE), ind_trials, 1);
	MN_grid_width = M/SM_CHUNK_SIZE;

	//NxM total threads with SM_CHUNK_SIZE^2 threads per 1D block
	MN_threads = dim3(SM_CHUNK_SIZE*SM_CHUNK_SIZE, 1, 1);
	MN_grid = dim3(M*N/(SM_CHUNK_SIZE*SM_CHUNK_SIZE), ind_trials, 1);
}

/**
 *	learn_step function
 */
template <LEARNING learning, DYNAMICS dynamics, PROFILING profiling>
void learn_step(int Tr, float* total_reward, float* total_error)
{
	

	//global learn step variables
	for (int i=0; i<ind_trials; i++)
	{
		total_reward[i] = 0;
		total_error[i] = 0;
	}

	srand ( (unsigned int)time(NULL) );
	for (int i=0;i<Tr*ind_trials;i++)
	{
		if (moving_target)
			h_In_Deg[i] = ((float)rand()/RAND_MAX)*2.0f*PI;
		else
			h_In_Deg[i] = static_target;
	}
	cutilSafeCall( cudaMemcpy(d_In_Deg, h_In_Deg, Tr*sizeof(float)*ind_trials, cudaMemcpyHostToDevice) );


	//allocate data for theta
	float* theta = (float*)malloc(ind_trials*sizeof(float));
	float* rew = (float*)malloc(ind_trials*sizeof(float));
	int* decision_offset = (int*)malloc(ind_trials*sizeof(int));
	int* target_offset = (int*)malloc(ind_trials*sizeof(int));

	for (int n=0; n<Tr; n++)
	{


		//set initial trial specific data
		cutilSafeCall( cudaMemset(d_Y, 0, T*N*sizeof(float)*ind_trials) );
		resetMembranePotential<<<NT_grid, NT_threads>>>(d_u, d_u_out);
		
		//calculate place cell distance and activations and output
		poissonNeuronSimulation<<<m_grid, m_threads>>>(n, Tr, d_X, d_Input, d_epsp, d_In_Deg, d_randSeeds, mn);
		cutilCheckMsg("Error in kernel\n");

		//for T trails (must be performed serially as lateral activation use data from t-1)
		for (int t=1; t<T; t++)
		{
			//calculate the place cell activations
			placeCellSpikePropagation<<<mN_grid, mN_threads>>>(d_W, d_Input+(t*M), d_input_pot);
			cutilCheckMsg("Error in kernel\n");

			//check if we are using dynamic simulation for lateral connections (if so calculate them)
			if (dynamics == DYN_SYS){
				//calculate the action cell lateral interactions
				actionCellLateralSpikePropagation<<<nN_grid, nN_threads>>>(d_Wr, d_Y+((t-1)*N), d_lateral_pot);
				cutilCheckMsg("Error in kernel\n");
			}

			//calculate the action cell spikes
			integrateAndFireNeuronSimulation<dynamics><<<n_grid, n_threads>>>(t, d_u, d_u_out, d_input_pot, d_lateral_pot, d_YProb, d_Y, d_randSeeds, mn);
			cutilCheckMsg("Error in kernel\n");

		
			//swap output for input
			float* d_u_temp;
			d_u_temp = d_u;
			d_u = d_u_out;
			d_u_out = d_u_temp;

		}
		

		//transpose Y
		transpose<N, T><<<NT_matrix_grid, NT_matrix_threads>>>(d_Y, d_Yt, NT_grid_width);
		cutilCheckMsg("Error in kernel\n");

		//calculate average angle of Y (for each N) across all ind trials N*ind_trials total parallel reductions
		reduceMultipleArrays<float>(T, d_Yt, d_Y_sum, N*ind_trials);
		cutilCheckMsg("Error in kernel\n");

		//calculate output angle components
		calculatePopulationVector<<<n_grid, n_threads>>>(d_Y_sum, d_out_x, d_out_y);
		cutilCheckMsg("Error in kernel\n");

		//sum output components
		reduceMultipleArrays<float>(N, d_out_x, d_out_x_sum, ind_trials);
		cutilCheckMsg("Error in kernel\n");
		reduceMultipleArrays<float>(N, d_out_y, d_out_y_sum, ind_trials);
		cutilCheckMsg("Error in kernel\n");


		//calculate angle by reading back sum totals to the CPU for each ind trial
		for (int i=0; i<ind_trials; i++)
		{
			int configuration_offset = pow2mod(i, ind_configs);
			float h_out_x_sum = 0;
			float h_out_y_sum = 0;
			int d_out_sum_offset = N*i;
			int d_In_Deg_offset = Tr*i;
			cutilSafeCall( cudaMemcpy( &h_out_x_sum, d_out_x_sum+d_out_sum_offset, sizeof(float), cudaMemcpyDeviceToHost) );
			cutilSafeCall( cudaMemcpy( &h_out_y_sum, d_out_y_sum+d_out_sum_offset, sizeof(float), cudaMemcpyDeviceToHost) );

			//calculate theta
			theta[i] = py_modf(atan2(h_out_x_sum, h_out_y_sum), 2.0f*PI);

			//calculate reward
			rew[i] = reward(abs(theta[i]-h_In_Deg[n+d_In_Deg_offset]), configuration_offset);

			//update total reward
			total_reward[i] += rew[i];
			total_error[i] += (fabs(h_In_Deg[n+d_In_Deg_offset] - theta[i]));

			//caluclate the descision and target offsets
			if (profiling == GRAPHING){
				decision_offset[i] = (int)floorf(((theta[i]+PI)*N)/(2.0f*PI));
				target_offset[i] = (int)floorf(((h_In_Deg[n+d_In_Deg_offset]+PI)*N)/(2.0f*PI));
			}
		}

		//copy rewards to device
		CUDA_SAFE_CALL( cudaMemcpy( d_rew, rew, ind_trials*sizeof(float), cudaMemcpyHostToDevice));
		
		//copy descision and target offsets to the device
		if (profiling == GRAPHING){
			CUDA_SAFE_CALL( cudaMemcpy( d_decision_offset, decision_offset, ind_trials*sizeof(int), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL( cudaMemcpy( d_target_offset, target_offset, ind_trials*sizeof(int), cudaMemcpyHostToDevice));
		}
		
		/* only perform the following if learning or calculating the gradiant graph*/
		if ((learning == UPDATE_LEARNING_WEIGHTS)||(profiling == GRAPHING))
		{
			//transpose rho
			transpose<N,T><<<NT_matrix_grid, NT_matrix_threads>>>(d_YProb, d_YProbt, NT_grid_width);
			cutilCheckMsg("Error in kernel\n");

			//calculate gradiant
			calculateGradiant<<<MN_matrix_grid, MN_matrix_threads>>>(MN_grid_width, d_Yt, d_YProbt, d_epsp, d_Grad);
			cutilCheckMsg("Error in kernel\n");

			if (profiling == GRAPHING)
			{
				//update the gradiant and deltaW sum totals
				updateGradiantAnalysis<<<MN_grid, MN_threads>>>(d_decision_offset, d_target_offset, d_Grad, d_rew, d_sumGrad, d_sumGrad_out, d_sumDeltaW, d_sumDeltaW_out); 
				cutilCheckMsg("Error in kernel\n");
				//swap sumGrad input and output
				float* d_sumGrad_temp;
				d_sumGrad_temp = d_sumGrad;
				d_sumGrad = d_sumGrad_out;
				d_sumGrad_out = d_sumGrad_temp;
				//swap deltaW input and output
				float* d_sumDeltaW_temp;
				d_sumDeltaW_temp = d_sumDeltaW;
				d_sumDeltaW = d_sumDeltaW_out;
				d_sumDeltaW_out = d_sumDeltaW_temp;

			}
			//no learning if we are calculating the gradiant graph
			else if (learning == UPDATE_LEARNING_WEIGHTS)
			{
				//update learning weights
				updateLearningWeights<<<MN_grid, MN_threads>>>(d_W, d_rew, d_Grad, d_W_out);
				cutilCheckMsg("Error in kernel\n");

				if (APPLY_NOISE)
				{
					//apply noise (this will also swap input and output)
					applyNoise<<<m_grid, m_threads>>>(d_W_out, d_W, d_randSeeds, mn);
					cutilCheckMsg("Error in kernel\n");
				}
				else{
					//swap the input and output pointer
					float* d_W_temp;	//used to swap input and output
					d_W_temp = d_W;
					d_W = d_W_out;
					d_W_out = d_W_temp;
				}
			}
		}
	}

	//cleanup theta
	free(theta);
	free(rew);
	free(decision_offset);
	free(target_offset);
}



void graphAnalysisDataToHost()
{
	printf("Copying graph analysis data from device to host\n");
	cutilSafeCall( cudaMemcpy(h_sumGrad, d_sumGrad, N*M*sizeof(float)*ind_trials, cudaMemcpyDeviceToHost) );
	cutilSafeCall( cudaMemcpy(h_sumDeltaW, d_sumDeltaW, N*M*sizeof(float)*ind_trials, cudaMemcpyDeviceToHost) );
}

void weightsToHost(){
	printf("Copying weight data from device to host\n");
	cutilSafeCall( cudaMemcpy(h_W, d_W, N*M*sizeof(float)*ind_trials, cudaMemcpyDeviceToHost) );
}


/** learn_curve
 * Runs the simulation
 */
template<PROFILING profiling, DYNAMICS dynamics> 
void learn_curve() 
{

	//init model
	copyParametersToDevice();
	initLearnStepData();
	calculateGridBlockSizes();
	if ((profiling == SIMULATION_EXTENDED_ANALYSIS)||(profiling == GRAPHING)){
		initGraphAnalysis();
		printf("Analysis mode....\n");
	}

	//start a timer
    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

	//allocate arrays for reward data
	float* total_analysis_rewards = (float*) malloc(ind_trials*no_intval*sizeof(float));
	float* total_analysis_errors = (float*) malloc(ind_trials*no_intval*sizeof(float));
	float* total_reward = (float*) malloc(ind_trials*sizeof(float));
	float* total_error = (float*) malloc(ind_trials*sizeof(float));

	//calculate the number of learn trials (scale no trials for no dyn)
	int learn_trials = 0;
	if (dynamics == DYN_SYS)
		learn_trials = learn_trials_dyn;
	else //NO_DYN_SYS
		learn_trials = learn_trials_no_dyn;

	int simulation_analysis_trials;
	if (profiling == SIMULATION)
		simulation_analysis_trials = analysis_trials;
	else //SIMULATION_EXTENDED_ANALYSIS
		simulation_analysis_trials = gradiant_analsyis_trials;

	
	//PROFILE
	if(profiling == PROFILE_ONLY)
	{
		if (dynamics==DYN_SYS)
			printf("Running dyn sys profiling:\n 1 step\n single trial\n %i ind configs\n %i ind trials per config\n", ind_configs, trials_per_config);
		if (dynamics==NO_DYN_SYS)
			printf("Running no dyn sys profiling:\n 1 step\n single trial\n %i ind configs\n %i ind trials per config\n", ind_configs, trials_per_config);

		learn_step<UPDATE_LEARNING_WEIGHTS, dynamics, PROFILE_ONLY>(1, total_reward, total_error);
	}
	//CREATE GRADIANT GRAPH: no learning
	else if(profiling == GRAPHING)
	{
		if (dynamics==DYN_SYS)
			printf("Running dyn sys graph analysis:\n 1 step\n %i trials\n %i ind configs\n %i ind trials per config\n", gradiant_analsyis_trials, ind_configs, trials_per_config);
		if (dynamics==NO_DYN_SYS)
			printf("Running no dyn sys graph analysis:\n 1 step\n %i trials\n %i ind configs\n %i ind trials per config\n", gradiant_analsyis_trials, ind_configs, trials_per_config);

		//reset values
		resetGraphAnalysis();

		//perform simulation
		learn_step<ANAYLSYS_ONLY, dynamics, GRAPHING>(gradiant_analsyis_trials, total_reward, total_error);

		//copy graph analysis dat to host
		graphAnalysisDataToHost();

		//output graphs
		printf("Saving graph analysis data...\n");
		saveGraphAnalysisData<dynamics>(h_sumGrad, h_sumDeltaW);
	}
	//SIMULATION
	else	
	{
		if (dynamics==DYN_SYS)
			printf("Running dyn sys simulation:\n %i steps\n %i analysis trials\n %i learning trials\n %i ind configs\n %i ind trials per config\n", no_intval, simulation_analysis_trials, learn_trials, ind_configs, trials_per_config);
		if (dynamics==NO_DYN_SYS)
			printf("Running no dyn sys simulation:\n %i steps\n %i analysis trials\n %i learning trials\n %i ind configs\n %i ind trials per config\n", no_intval, simulation_analysis_trials, learn_trials, ind_configs, trials_per_config);
		
		for (int m=0; m<no_intval; m++)
		{
			printf("Starting Step %i of %i\n", (m+1), no_intval);
			float* no_intval_reward = &total_analysis_rewards[m*ind_trials];
			float* no_intval_error = &total_analysis_errors[m*ind_trials];

			//Perform analsyis
			if (profiling == SIMULATION)
			{
				printf("Stage 1: Performing analysis...\n");
				//perform simulation analysis
				learn_step<ANAYLSYS_ONLY, dynamics, SIMULATION>(analysis_trials, no_intval_reward, no_intval_error);
			}
			else //SIMULATION_EXTENDED_ANALYSIS
			{
				printf("Performing extended analysis...\n");
				
				//reset values
				resetGraphAnalysis();
				
				//perform extended simulation analysis
				learn_step<ANAYLSYS_ONLY, dynamics, GRAPHING>(gradiant_analsyis_trials, no_intval_reward, no_intval_error);

				//copy graph analysis data to host and produce the graph data for the current step
				graphAnalysisDataToHost();
				printf("Saving graph analysis data for step %i...\n", (m+1));
				saveGraphAnalysisData<dynamics>(h_sumGrad, h_sumDeltaW, (m+1));
			}

			//Print analsyis
			for(int j=0; j<ind_configs; j++)
			{
				for(int i=0; i<trials_per_config; i++)
				{
					printf("Step %i, Config %i, Ind trial no: %i: Av Reward is %f, Av Error %f\n", (m+1), (j+1), (i+1), no_intval_reward[j+(i*ind_configs)]/(float)simulation_analysis_trials ,no_intval_error[j+(i*ind_configs)]/(float)simulation_analysis_trials);
				}
			}

			//perform learning
			printf("Performing learn step...\n", (m+1), no_intval);
			learn_step<UPDATE_LEARNING_WEIGHTS, dynamics, SIMULATION>(learn_trials, total_reward, total_error);

			//for extended analysis print weights after each step
			if((profiling==SIMULATION_EXTENDED_ANALYSIS)&&(printWeightPlot))
			{
				weightsToHost();
				printf("Saving weight data...\n");
				saveWeightData<dynamics>(h_W, (m+1));
			}
		}
	}
	
	//stop the timer
    cutilCheckError( cutStopTimer( timer));
	printf("Simulation complete\n");
    printf("Processing time: %f (seconds)\n\n", cutGetTimerValue( timer)/1000.0f);
    cutilCheckError( cutDeleteTimer( timer));

	//output data to files ifn ot profiling
	if ((profiling == SIMULATION)||(profiling == SIMULATION_EXTENDED_ANALYSIS)){
		printf("Saving learn curve data...\n");
		saveLearnCurveData<dynamics>(total_analysis_rewards);
		saveErrorCurveData<dynamics>(total_analysis_errors);
	}

	//for simulation mode print weights once at end of simulation step
	if((profiling==SIMULATION)&&(printWeightPlot))
	{
		weightsToHost();
		printf("Saving weight data...\n");
		saveWeightData<dynamics>(h_W);
	}



	//free array for reward data
	free(total_reward);
	free(total_analysis_rewards);
	free(total_analysis_errors);

	//cleanup
	cleanupLearnStep();
	if ((profiling == SIMULATION_EXTENDED_ANALYSIS)||(profiling == GRAPHING)){
		cleanupGraphAnalysis();
	}
}


////////////////////////////////////////////////////////////////////////////////
// Program main

int
main( int argc, char** argv) 
{
	PROFILING profile;
	bool dyn_sys;
	bool no_dyn_sys;

	//init the device using command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

	//profile
	profile = SIMULATION;
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "profile") )
		profile = PROFILE_ONLY;
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "graph_analysis") )
		profile = GRAPHING;
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "extended_analysis") )
		profile = SIMULATION_EXTENDED_ANALYSIS;
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "print_weight_plot") )
		printWeightPlot = true;

	//check for invalid use of pwint weight plot
	if ((profile == GRAPHING)&&(printWeightPlot)){
		printf("Cannot use print_weight_plot argument with graph_analysis.\n");
		cudaThreadExit();
		cutilExit(argc, argv);
		exit(0);
	}

	
	//dyn sys (default perform both dynamics and no dynamics sequentially
	dyn_sys = true;
	no_dyn_sys = true;
	//dynamic system only
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "dyn_sys") )
		no_dyn_sys = false;
	else if( cutCheckCmdLineFlag(argc, (const char**)argv, "no_dyn_sys") )
		dyn_sys = false;

	//perform dynamic system simulation
	if (dyn_sys)
	{
		if (profile == PROFILE_ONLY)
			learn_curve<PROFILE_ONLY, DYN_SYS>();
		else if (profile == GRAPHING)
			learn_curve<GRAPHING, DYN_SYS>();
		else if (profile == SIMULATION_EXTENDED_ANALYSIS)
			learn_curve<SIMULATION_EXTENDED_ANALYSIS, DYN_SYS>();
		//SIMULTATION
		else	
			learn_curve<SIMULATION, DYN_SYS>();
	}

	//perform non dydnamic system simulation
	if(no_dyn_sys)
	{
		if (profile == PROFILE_ONLY)
			learn_curve<PROFILE_ONLY, NO_DYN_SYS>();
		else if (profile == GRAPHING)
			learn_curve<GRAPHING, NO_DYN_SYS>();
		else if (profile == SIMULATION_EXTENDED_ANALYSIS)
			learn_curve<SIMULATION_EXTENDED_ANALYSIS, NO_DYN_SYS>();
		//SIMULTATION
		else
			learn_curve<SIMULATION, NO_DYN_SYS>();
	}

	//if simulating
	if ((profile == SIMULATION)||(profile ==SIMULATION_EXTENDED_ANALYSIS))
	{
		printf("Creating learn curve graphs...\n");
		createLearnCurveGraph(dyn_sys, no_dyn_sys);
		createErrorCurveGraph(dyn_sys, no_dyn_sys);

		if (profile == SIMULATION_EXTENDED_ANALYSIS)
		{
			for (int m=0; m<no_intval; m++)
			{
				printf("Creating analysis graphs for step %i...\n", (m+1));
				createAnalysisGraphs(dyn_sys, no_dyn_sys, (m+1));

				//create a weight graph for each step
				if (printWeightPlot)
				{
					printf("Creating 3d weights plot for step %i...\n", (m+1));
					createWeightGraphs(dyn_sys, no_dyn_sys, (m+1));
				}

			}
		}
		else //SIMULATION
		{
			//create a single weight plot graph
			if (printWeightPlot)
			{
				printf("Creating 3d weights plot...\n");
				createWeightGraphs(dyn_sys, no_dyn_sys);
			}
		}
	}
	//if performing graph analysis
	else if (profile == GRAPHING)
	{
		printf("Creating analysis graphs...\n");
		createAnalysisGraphs(dyn_sys, no_dyn_sys);
	}


	

	cudaThreadExit();

	//this will pause the window if we are not profiling
	if(profile != PROFILE_ONLY)
		cutilExit(argc, argv);
}
