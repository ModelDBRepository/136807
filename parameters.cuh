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
This CUDA header file contains the modelling parameters which are set
by default to the values used within the accompanying paper. Parameters 
prefixed with "h" are indicated as host copies of variables which are 
copied to the GPU device variables (declared using "__constant__ float 
variable_name") by the function  copyParametersToDevice in model.cu.
*/

#ifndef _BANDIT_PARAMTERS_H
#define _BANDIT_PARAMTERS_H

#include "model.h"

//Used to control if the target is assigned a random position for each learn steps (if true the system will learn much faster)
#define moving_target true
//If the above value is false then the static target value indicates the constant target value
#define static_target PI
//Used to indicate that noise should be applied to the system (using the h_w0 value)
#define APPLY_NOISE false

//Used to indicate that the neurons demonstrate a refractory period after firing
#define REFACTORY true

//Used to specify the reward shape of the reward function (GUASSIAN or BOX)
static const REWARD_FUNCTION reward_func = GAUSSIAN;

//Learn curve parameters
#define no_intval 10							//number of learning itterations
#define analysis_trials 128						//number of trials during analysis
#define gradiant_analsyis_trials 1024			//number of trials used for gradiant graph anaylsis
#define learn_trials_dyn 512					//number of trials during learning
#define learn_trials_no_dyn 512					//number of trials during learning

//Output directory for GNU plot files
//IMPORTANT: output directory must already exist
static const char* config_output_dir = "output";

//Trial paramaters
#define ind_configs 1								//number of indipendant configurations (must be power of 2)
#define trials_per_config 16						//ind trials per configuration (must be power of 2)
#define ind_trials ind_configs*trials_per_config	//total number of indepantant trials

/* The value of 'ind_configs' determines how many parameter values are expected for each parameter defined below this point */

//output filename prefix
static const char* config_output_prefix[] = {"init_params"}; 

//Model parameters (Host)
static const float sig_p[ind_configs] = {7.0f};					//CONNECTIVITY: length scales of  mexican head connectivity
static const float rsc[ind_configs] = {0.325f};	//0.325			//CONNECTIVITY: scaling of recurrent connectivity
static const float w_E[ind_configs] = {7.0f};					//EXCITEMENT: excitatiory weights
static const float sigma_R[ind_configs] = {PI/2.0f};			//REWARD: standard deviation of Guassion reward function
static const float sigma_R_box[ind_configs] = {0.0f};			//REWARD: reward interval of Box reward function


//Model Parameters (Device) - Host Copy
static const float h_xsc[ind_configs] = {0.35f}; 				//CONNECTIVITY: scaling of input activity
static const float h_beta[ind_configs] = {0.2f}; //0.2			//OVERLAP: size of overlap for receptive fields
static const float h_u0[ind_configs] = {-2.0f};					//NOISE: constant offset of membrane potential
static const float h_w0[ind_configs] = {0.0000f};				//NOISE: direct noise of dw (only applied if APPLY_NOISE == true)
static const float h_eta[ind_configs]    = {0.005f};			//LEARNING: learning rate
static const float h_baseline[ind_configs] = {0.0f};			//LEARNING: learn rate baseline

//Model Parameters for Spiking Neuron
static const float h_tau[ind_configs]= {10.0f};					//Membrane constant (ms)
static const float h_lambda[ind_configs]= {1.0f-(dt/h_tau[0])};	//Variable actually used in the calculations
static const float h_urest[ind_configs]= {-70.0f};				//Rest voltage (mV)
static const float h_threshold[ind_configs]= {-50.0f};			//Firing threshold voltage mV
static const float h_du[ind_configs]   = {5.0f};				//(mV)
static const float h_rho0[ind_configs]= {1.0f};					//(1/ms)
static const float h_ref[ind_configs]= {5.0f};					//(1/ms)

/* Model Parameters (Device) - These can not be modifed directly. Instead modify the host versions.  */ 

__constant__ float beta[ind_configs];		
__constant__ float xsc[ind_configs];		
__constant__ float u0[ind_configs];		
__constant__ float w0[ind_configs];			
__constant__ float eta[ind_configs];	
__constant__ float baseline[ind_configs];

__constant__ float tau[ind_configs];
__constant__ float lambda[ind_configs];
__constant__ float urest[ind_configs];
__constant__ float threshold[ind_configs];
__constant__ float du[ind_configs];
__constant__ float rho0[ind_configs];
__constant__ float ref[ind_configs];



#endif _BANDIT_PARAMTERS_H