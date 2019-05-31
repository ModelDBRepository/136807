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
This header file contains any modelling parameters which are defined 
not as variables but as model specific values and hence cannot be varied 
across individual simulation configurations. Most notably these are the 
neuron populations sizes and reference to time and the time integral dt. 
In addition to this a number of enumerations used throughout the model 
are defined here.
*/


#ifndef _BANDIT_H
#define _BANDIT_H


//Definitions
#define PI 3.1415f				//PI
#define dt 1.0f					//dt
#define N 256					//population size of action cells (must a multiple of SM_CHUNK_SIZE and devisor of THREADS_PER_BLOCK)
#define M 256					//population size of place cells (must a multiple of SM_CHUNK_SIZE and devisor of THREADS_PER_BLOCK)
#define time_period 128			//total time period (must be a power of 2)
#define T 128					//number of discrete time intervals i.e. time_period/T


//reward function
enum REWARD_FUNCTION
{
	GAUSSIAN,
	BOX
};

//model enumeration definitions
enum PROFILING
{
	PROFILE_ONLY,
	SIMULATION,
	SIMULATION_EXTENDED_ANALYSIS,
	GRAPHING
};

enum LEARNING
{
	ANAYLSYS_ONLY,
	UPDATE_LEARNING_WEIGHTS
};

enum DYNAMICS
{
	NO_DYN_SYS,
	DYN_SYS
};



#endif