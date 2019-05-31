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
This CUDA source file contains the function definitions used for creating data 
plots and generating GPU plot scripts.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <time.h>
#include <float.h>
#include <cutil_inline.h>

#include "output.cuh"

template<DYNAMICS dynamics>
void printConfigurationToFile(FILE* stream, int config_no)
{
	time_t rawtime;
	struct tm * timeinfo;
			
	//output configuration info
	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	fprintf(stream, "# Output from GPU model: %s", asctime(timeinfo));
	fprintf(stream, "# Configuration number (%i)\n", config_no);
	if (dynamics == DYN_SYS)
		fprintf(stream, "# Simulation using Dynamical Lateral Connections (%i)\n", config_no);
	else //NO_DYN_SYS
		fprintf(stream, "# Simulation NOT using Dynamical Lateral Connections (%i)\n", config_no);
	if (moving_target)
		fprintf(stream, "# Random moving target between trials\n");
	else
		fprintf(stream, "# Static moving target (%f) between trials\n", static_target);
	if (REFACTORY)
		fprintf(stream, "# Refactory period for spiking neurons\n");
	else
		fprintf(stream, "# No refactory period for spiking neurons\n");

	fprintf(stream, "# \n");
	fprintf(stream, "# Global Values ************************\n");
	fprintf(stream, "# N = %i \n", N);
	fprintf(stream, "# M = %i \n", M);
	fprintf(stream, "# T = %i \n", T);

	fprintf(stream, "# Learning Step Values ************************\n");
	fprintf(stream, "# no_intval = %i \n", no_intval);
	fprintf(stream, "# analysis_trials = %i \n", analysis_trials);
	fprintf(stream, "# gradiant_analysis_trials = %i \n", gradiant_analsyis_trials);
	fprintf(stream, "# learn_trials_dyn_sys = %i \n", learn_trials_dyn);
	fprintf(stream, "# learn_trials_no_dyn_sys = %i \n", learn_trials_no_dyn);
	fprintf(stream, "# trials_per_config = %i \n", trials_per_config);

	fprintf(stream, "# Simulation Parameters ****************\n");
	fprintf(stream, "# sig_p = %f \n", sig_p[config_no]);
	fprintf(stream, "# rsc = %f \n", rsc[config_no]);
	fprintf(stream, "# w_E = %f \n", w_E[config_no]);
	if (reward_func == GAUSSIAN){
		fprintf(stream, "# GAUSSIAN REWARD FUNCTION\n");
		fprintf(stream, "# sigma_R = %f \n", sigma_R[config_no]);
	}else{	//BOX
		fprintf(stream, "# BOX REWARD FUNCTION\n");
		fprintf(stream, "# sigma_R_box = %f \n", sigma_R_box[config_no]);
	}
	fprintf(stream, "# beta = %f \n", h_beta[config_no]);
	fprintf(stream, "# xsc = %f \n", h_xsc[config_no]);
	fprintf(stream, "# w0 = %f \n", h_w0[config_no]);
	fprintf(stream, "# u0 = %f \n", h_u0[config_no]);
	fprintf(stream, "# eta = %f \n", h_eta[config_no]);
	fprintf(stream, "# baseline = %f \n", h_baseline[config_no]);
	fprintf(stream, "# \n\n\n");

	fprintf(stream, "# Spiking Neuron Parameters ****************\n");
	fprintf(stream, "# dt = %f \n", dt);
	fprintf(stream, "# tau = %f \n", h_tau[config_no]);
	fprintf(stream, "# lambda = %f \n", h_lambda[config_no]);
	fprintf(stream, "# urest = %f \n", h_urest[config_no]);
	fprintf(stream, "# threshold = %f \n", h_threshold[config_no]);
	fprintf(stream, "# du = %f \n", h_du[config_no]);
	fprintf(stream, "# rho0 = %f \n", h_rho0[config_no]);
	fprintf(stream, "# ref = %f \n", h_ref[config_no]);

}

template<DYNAMICS dynamics>
void saveLearnCurveData(float* total_analysis_rewards)
{
	for (int f = 0; f< ind_configs; f++)
	{
		char filename[128];
		FILE* ar;
		if (dynamics == DYN_SYS)
			sprintf(filename, "%s/%s_average_reward(dyn_sys).dat", config_output_dir, config_output_prefix[f]);
		else //NO_DYN_SYS
			sprintf(filename, "%s/%s_average_reward(no_dyn_sys).dat", config_output_dir, config_output_prefix[f]);
		ar = fopen(filename, "w");
		if (ar == NULL){
			printf("Error: Can't open output file %s!\n", filename);
			exit(0);
		}

		printConfigurationToFile<dynamics>(ar, f);

		fprintf(ar, "# Average Reward Data ******************\n");
		fprintf(ar, "# Step\tMEAN_REWARD\tSTD DEVIATION\n");
		
		float average_total = 0.0f;
		float sd_total = 0.0f;
		//calculate averages
		for (int m=0; m<no_intval; m++)
		{
			float average_reward = 0;
			float std_dev = 0;

			//For each step of each config calculate the average accross the ind trials
			for(int i=0; i<trials_per_config; i++)
			{
				float* no_intval_reward = &total_analysis_rewards[m*ind_trials];
				float reward = no_intval_reward[f+(i*ind_configs)]/(float)analysis_trials;
				average_reward += reward;
			}
			average_reward /= (float)trials_per_config;

			//For each step of each config calculate the SD accross the ind trials
			for(int i=0; i<trials_per_config; i++)
			{
				float* no_intval_reward = &total_analysis_rewards[m*ind_trials];
				float reward = no_intval_reward[f+(i*ind_configs)]/(float)analysis_trials;
				std_dev += (average_reward - reward)*(average_reward - reward);
			}
			std_dev /= (float)trials_per_config;
			std_dev = sqrtf(std_dev);


			//print to data file with error bars
			fprintf(ar, "%i\t%f\t%f\n", m, average_reward, std_dev);

			average_total += average_reward;
			sd_total += std_dev;
		}

		average_total /= no_intval;
		sd_total /= no_intval;
		fprintf(ar, "# average over steps\n");
		fprintf(ar, "# %f\t%f\n", average_total, sd_total);

		//close the file
		fclose(ar);
	}
}

template<DYNAMICS dynamics>
void saveErrorCurveData(float* total_reward_errors)
{
	for (int f = 0; f< ind_configs; f++)
	{
		char filename[128];
		FILE* ar;
		if (dynamics == DYN_SYS)
			sprintf(filename, "%s/%s_reward_error(dyn_sys).dat", config_output_dir, config_output_prefix[f]);
		else //NO_DYN_SYS
			sprintf(filename, "%s/%s_reward_error(no_dyn_sys).dat", config_output_dir, config_output_prefix[f]);
		ar = fopen(filename, "w");
		if (ar == NULL){
			printf("Error: Can't open output file %s!\n", filename);
			exit(0);
		}

		printConfigurationToFile<dynamics>(ar, f);

		fprintf(ar, "# Reward Error Data ******************\n");
		fprintf(ar, "# Step\tMEAN_REWARD\tSTD DEVIATION\n");
		
		float error_total = 0.0f;
		float sd_total = 0.0f;
		//calculate averages
		for (int m=0; m<no_intval; m++)
		{
			float average_error = 0;
			float std_dev = 0;

			//For each step of each config calculate the average accross the ind trials
			for(int i=0; i<trials_per_config; i++)
			{
				float* no_intval_reward = &total_reward_errors[m*ind_trials];
				float error = no_intval_reward[f+(i*ind_configs)]/(float)analysis_trials;
				average_error += error;
			}
			average_error /= (float)trials_per_config;

			//For each step of each config calculate the SD accross the ind trials
			for(int i=0; i<trials_per_config; i++)
			{
				float* no_intval_reward = &total_reward_errors[m*ind_trials];
				float error = no_intval_reward[f+(i*ind_configs)]/(float)analysis_trials;
				std_dev += (average_error - error)*(average_error - error);
			}
			std_dev /= (float)trials_per_config;
			std_dev = sqrtf(std_dev);


			//print to data file with error bars
			fprintf(ar, "%i\t%f\t%f\n", m, average_error, std_dev);

			error_total += average_error;
			sd_total += std_dev;
		}

		error_total /= no_intval;
		sd_total /= no_intval;
		fprintf(ar, "# average over steps\n");
		fprintf(ar, "# %f\t%f\n", error_total, sd_total);

		//close the file
		fclose(ar);
	}
}

void createLearnCurveGraph(bool dyn_sys, bool no_dyn_sys)
{
	for (int f = 0; f< ind_configs; f++)
	{
		char filename[128];
		char dyn_sys_filename[128];
		char no_dyn_sys_filename[128];
		time_t rawtime;
		struct tm * timeinfo;
		FILE* ar;

		//file names
		sprintf(filename, "%s/%s_reward_error.plt", config_output_dir, config_output_prefix[f]);
		ar = fopen(filename, "w");
		if (ar == NULL){
			printf("Error: Can't open output file %s!\n", filename);
			exit(0);
		}

		//files used for graph input
		sprintf(dyn_sys_filename, "%s_reward_error(dyn_sys).dat", config_output_prefix[f]);
		sprintf(no_dyn_sys_filename, "%s_reward_error(no_dyn_sys).dat", config_output_prefix[f]);

		//output file header information
		time ( &rawtime );
		timeinfo = localtime ( &rawtime );
		fprintf(ar, "# Output from GPU model: %s", asctime(timeinfo));
		fprintf(ar, "# GNUPlot Script");
		fprintf(ar, "set title 'Average Reward Error Plot for %s'\n", config_output_prefix[f]);
		fprintf(ar, "set xlabel 'learning steps'\n");
		fprintf(ar, "set ylabel 'reward error'\n");

		//create plot
		if (dyn_sys && no_dyn_sys){
			fprintf(ar, "plot '%s' with lines, '%s'  with yerrorbars title 'SD accross ind trials', '%s' with lines, '%s' with yerrorbars title 'SD accross ind trials'",	dyn_sys_filename, dyn_sys_filename, no_dyn_sys_filename, no_dyn_sys_filename);
		}
		else
		{
			if(dyn_sys)
				fprintf(ar, "plot '%s' with lines, '%s' with yerrorbars title 'SD accross ind trials",	dyn_sys_filename, dyn_sys_filename, no_dyn_sys_filename, no_dyn_sys_filename);
			if(no_dyn_sys)
				fprintf(ar, "plot '%s' with lines, '%s' with yerrorbars title 'SD accross ind trials", no_dyn_sys_filename, no_dyn_sys_filename);
		}
		fclose(ar);
	}

}


void createErrorCurveGraph(bool dyn_sys, bool no_dyn_sys)
{
	for (int f = 0; f< ind_configs; f++)
	{
		char filename[128];
		char dyn_sys_filename[128];
		char no_dyn_sys_filename[128];
		time_t rawtime;
		struct tm * timeinfo;
		FILE* ar;

		//file names
		sprintf(filename, "%s/%s_average_reward.plt", config_output_dir, config_output_prefix[f]);
		ar = fopen(filename, "w");
		if (ar == NULL){
			printf("Error: Can't open output file %s!\n", filename);
			exit(0);
		}

		//files used for graph input
		sprintf(dyn_sys_filename, "%s_average_reward(dyn_sys).dat", config_output_prefix[f]);
		sprintf(no_dyn_sys_filename, "%s_average_reward(no_dyn_sys).dat", config_output_prefix[f]);

		//output file header information
		time ( &rawtime );
		timeinfo = localtime ( &rawtime );
		fprintf(ar, "# Output from GPU model: %s", asctime(timeinfo));
		fprintf(ar, "# GNUPlot Script");
		fprintf(ar, "set title 'Average Reward Plot for %s'\n", config_output_prefix[f]);
		fprintf(ar, "set xlabel 'learning steps'\n");
		fprintf(ar, "set ylabel 'average reward'\n");

		//create plot
		if (dyn_sys && no_dyn_sys){
			fprintf(ar, "plot '%s' with lines, '%s'  with yerrorbars title 'SD accross ind trials', '%s' with lines, '%s' with yerrorbars title 'SD accross ind trials'",	dyn_sys_filename, dyn_sys_filename, no_dyn_sys_filename, no_dyn_sys_filename);
		}
		else
		{
			if(dyn_sys)
				fprintf(ar, "plot '%s' with lines, '%s' with yerrorbars title 'SD accross ind trials",	dyn_sys_filename, dyn_sys_filename, no_dyn_sys_filename, no_dyn_sys_filename);
			if(no_dyn_sys)
				fprintf(ar, "plot '%s' with lines, '%s' with yerrorbars title 'SD accross ind trials", no_dyn_sys_filename, no_dyn_sys_filename);
		}
		fclose(ar);
	}

}

template<DYNAMICS dynamics>
void saveGraphAnalysisData(float* h_sumGrad, float* h_sumDeltaW, int step)
{
	char step_str[128] = "";
	if (step > 0)
	{
		sprintf(step_str, "_step_%i", step);
	}
	for (int f = 0; f< ind_configs; f++)
	{
		//create and open files for writing mean gradiant and mean deltaW
		char mG_filename[128];
		char dW_filename[128];
		FILE* mG;
		FILE* dW;
		if (dynamics == DYN_SYS){
			sprintf(mG_filename, "%s/%s_gradiant(dyn_sys)%s.dat", config_output_dir, config_output_prefix[f], step_str);
			sprintf(dW_filename, "%s/%s_deltaW(dyn_sys)%s.dat", config_output_dir, config_output_prefix[f], step_str);
		}
		else{ //NO_DYN_SYS
			sprintf(mG_filename, "%s/%s_gradiant(no_dyn_sys)%s.dat", config_output_dir, config_output_prefix[f], step_str);
			sprintf(dW_filename, "%s/%s_deltaW(no_dyn_sys)%s.dat", config_output_dir, config_output_prefix[f], step_str);
		}
		mG = fopen(mG_filename, "w");
		if (mG == NULL){
			printf("Error: Can't open output file %s!\n", mG_filename);
			exit(0);
		}
		dW = fopen(dW_filename, "w");
		if (dW == NULL){
			printf("Error: Can't open output file %s!\n", dW_filename);
			exit(0);
		}

		//output file header information
		printConfigurationToFile<dynamics>(mG, f);
		printConfigurationToFile<dynamics>(dW, f);
		
		//loop through action cells
		for (int n=0; n<N; n++)
		{
			float avr_grad = 0.0f;	
			float avr_deltaW = 0.0f;	
			//average for each neuron connection (xM) and over the indipendant trials
			for(int i=0; i<trials_per_config; i++)
			{
				//offset for each indipendant trial
				int trial_offset = (f+(i*ind_configs))*M*N;
				for (int m=0; m<M; m++)
				{
					avr_grad += h_sumGrad[(n*M) + m + trial_offset];
					avr_deltaW += h_sumDeltaW[(n*M) + m + trial_offset];
				}
			}
			//average
			avr_grad /= M*trials_per_config;
			avr_deltaW /= M*trials_per_config;
			
			//print to data files
			fprintf(mG, "%i\t%f\n", n, avr_grad);
			fprintf(dW, "%i\t%f\n", n, avr_deltaW);
		}

		//close the files
		fclose(mG);
		fclose(dW);
	}
}

void createAnalysisGraphs(bool dyn_sys, bool no_dyn_sys, int step)
{
	char step_str[128] = "";
	if (step > 0)
	{
		sprintf(step_str, "_step_%i", step);
	}

	for (int f = 0; f< ind_configs; f++)
	{
		char mG_filename[128];
		char dW_filename[128];
		char dyn_sys_mG_filename[128];
		char no_dyn_sys_mG_filename[128];
		char dyn_sys_dW_filename[128];
		char no_dyn_sys_dW_filename[128];
		time_t rawtime;
		struct tm * timeinfo;
		FILE* mG;
		FILE* dW;

		//file names
		sprintf(mG_filename, "%s/%s_gradiant%s.plt", config_output_dir, config_output_prefix[f], step_str);
		sprintf(dW_filename, "%s/%s_deltaW%s.plt", config_output_dir, config_output_prefix[f], step_str);
		mG = fopen(mG_filename, "w");
		if (mG == NULL){
			printf("Error: Can't open output file %s!\n", mG_filename);
			exit(0);
		}
		dW = fopen(dW_filename, "w");
		if (dW == NULL){
			printf("Error: Can't open output file %s!\n", dW_filename);
			exit(0);
		}

		//files used for graph input
		sprintf(dyn_sys_mG_filename, "%s_gradiant(dyn_sys)%s.dat", config_output_prefix[f], step_str);
		sprintf(no_dyn_sys_mG_filename, "%s_gradiant(no_dyn_sys)%s.dat", config_output_prefix[f], step_str);
		sprintf(dyn_sys_dW_filename, "%s_deltaW(dyn_sys)%s.dat", config_output_prefix[f], step_str);
		sprintf(no_dyn_sys_dW_filename, "%s_deltaW(no_dyn_sys)%s.dat", config_output_prefix[f], step_str);

		//get time info
		time ( &rawtime );
		timeinfo = localtime ( &rawtime );

		//output file header infor for gradiant mean
		fprintf(mG, "# Output from GPU model: %s", asctime(timeinfo));
		fprintf(mG, "# GNUPlot Script");
		fprintf(mG, "# %i gradiant anyalysis trials", gradiant_analsyis_trials);
		fprintf(mG, "# %i indipendant trials per configuration", trials_per_config);
		fprintf(mG, "# Step %i of %i", step, no_intval);
		fprintf(mG, "set title 'Action Cells Mean Gradiant (before learning)%s'\n", config_output_prefix[f]);
		fprintf(mG, "set xlabel 'Action Cell Index (aligned by decision)'\n");
		fprintf(mG, "set ylabel 'mean gradiant'\n");

		//output file header infor for delta W
		fprintf(dW, "# Output from GPU model: %s", asctime(timeinfo));
		fprintf(dW, "# GNUPlot Script");
		fprintf(dW, "# %i gradiant anyalysis trials", gradiant_analsyis_trials);
		fprintf(dW, "# %i indipendant trials per configuration", trials_per_config);
		fprintf(dW, "set title 'Action Cells Delta W (before learning)%s'\n", config_output_prefix[f]);
		fprintf(dW, "set xlabel 'Action Cell Index (aligned by target angle and offset 180*)'\n");
		fprintf(dW, "set ylabel 'delta W'\n");

		//create plot
		if (dyn_sys && no_dyn_sys){
			fprintf(mG, "plot '%s' with lines, '%s' with lines",	dyn_sys_mG_filename, no_dyn_sys_mG_filename);
			fprintf(dW, "plot '%s' with lines, '%s' with lines",	dyn_sys_dW_filename, no_dyn_sys_dW_filename);
		}
		else
		{
			if(dyn_sys){
				fprintf(mG, "plot '%s' with lines", dyn_sys_mG_filename);
				fprintf(dW, "plot '%s' with lines", dyn_sys_dW_filename);
			}
			if(no_dyn_sys){
				fprintf(mG, "plot '%s' with lines", no_dyn_sys_mG_filename);
				fprintf(dW, "plot '%s' with lines", no_dyn_sys_dW_filename);
			}
		}

		//close files
		fclose(mG);
		fclose(dW);
	}

}


template<DYNAMICS dynamics>
void saveWeightData(float* h_W, int step)
{
char step_str[128] = "";
	if (step > 0)
	{
		sprintf(step_str, "_step_%i", step);
	}
	for (int f = 0; f< ind_configs; f++)
	{
		//create and open files for writing mean gradiant and mean deltaW
		char w_filename[128];
		FILE* w;
		if (dynamics == DYN_SYS){
			sprintf(w_filename, "%s/%s_W(dyn_sys)%s.dat", config_output_dir, config_output_prefix[f], step_str);
		}
		else{ //NO_DYN_SYS
			sprintf(w_filename, "%s/%s_W(no_dyn_sys)%s.dat", config_output_dir, config_output_prefix[f], step_str);
		}
		w = fopen(w_filename, "w");
		if (w == NULL){
			printf("Error: Can't open output file %s!\n", w_filename);
			exit(0);
		}

		//output file header information
		printConfigurationToFile<dynamics>(w, f);

		
		//loop through action cells
		for (int n=0; n<N; n++)
		{
			for (int m=0; m<M; m++)
			{
				float avr_w = 0;
				for(int i=0; i<trials_per_config; i++)
				{
					//offset for each indipendant trial
					int trial_offset = (f+(i*ind_configs))*M*N;

					avr_w += h_W[(n*M) + m + trial_offset];
				}
				avr_w /= trials_per_config;
				fprintf(w, "%i\t%i\t%f\n", n, m, avr_w);
			}
			fprintf(w, "\n");
		}

		//close the files
		fclose(w);
	}

}

void createWeightGraphs(bool dyn_sys, bool no_dyn_sys, int step)
{
	char step_str[128] = "";
	if (step > 0)
	{
		sprintf(step_str, "_step_%i", step);
	}

	for (int f = 0; f< ind_configs; f++)
	{
		char dyn_sys_w_plot_filename[128];
		char dyn_sys_w_filename[128];
		char no_dyn_sys_w_plot_filename[128];
		char no_dyn_sys_w_filename[128];
		time_t rawtime;
		struct tm * timeinfo;
		FILE* w_dyn_sys;
		FILE* w_no_dyn_sys;

		//get time info
		time ( &rawtime );
		timeinfo = localtime ( &rawtime );

		if(dyn_sys){

			//file names
			sprintf(dyn_sys_w_plot_filename, "%s/%s_W(dyn_sys)%s.plt", config_output_dir, config_output_prefix[f], step_str);
			w_dyn_sys = fopen(dyn_sys_w_plot_filename, "w");
			if (w_dyn_sys == NULL){
				printf("Error: Can't open output file %s!\n", dyn_sys_w_plot_filename);
				exit(0);
			}

			//files used for graph input
			sprintf(dyn_sys_w_filename, "%s_W(dyn_sys)%s.dat", config_output_prefix[f], step_str);

			//output file header infor for gradiant mean
			fprintf(w_dyn_sys, "# Output from GPU model: %s", asctime(timeinfo));
			fprintf(w_dyn_sys, "# GNUPlot Script");
			fprintf(w_dyn_sys, "# %i gradiant anyalysis trials", gradiant_analsyis_trials);
			fprintf(w_dyn_sys, "# %i indipendant trials per configuration", trials_per_config);
			fprintf(w_dyn_sys, "# Step %i of %i", step, no_intval);
			fprintf(w_dyn_sys, "set title 'Action Cell Weight: Step %i %s'\n", step, config_output_prefix[f]);
			fprintf(w_dyn_sys, "set xlabel 'N'\n");
			fprintf(w_dyn_sys, "set ylabel 'M'\n");	

			fprintf(w_dyn_sys, "plot '%s' with image", dyn_sys_w_filename);

			fclose(w_dyn_sys);

		}

		
		if(no_dyn_sys){
			sprintf(no_dyn_sys_w_plot_filename, "%s/%s_W(no_dyn_sys)%s.plt", config_output_dir, config_output_prefix[f], step_str);
			w_no_dyn_sys = fopen(no_dyn_sys_w_plot_filename, "w");
			if (w_no_dyn_sys == NULL){
				printf("Error: Can't open output file %s!\n", no_dyn_sys_w_plot_filename);
				exit(0);
			}

			sprintf(no_dyn_sys_w_filename, "%s_W(no_dyn_sys)%s.dat", config_output_prefix[f], step_str);

			fprintf(w_no_dyn_sys, "# Output from GPU model: %s", asctime(timeinfo));
			fprintf(w_no_dyn_sys, "# GNUPlot Script");
			fprintf(w_no_dyn_sys, "# %i gradiant anyalysis trials", gradiant_analsyis_trials);
			fprintf(w_no_dyn_sys, "# %i indipendant trials per configuration", trials_per_config);
			fprintf(w_no_dyn_sys, "# Step %i of %i", step, no_intval);
			fprintf(w_no_dyn_sys, "set title 'Action Cell Weight: Step %i %s'\n", step, config_output_prefix[f]);
			fprintf(w_no_dyn_sys, "set xlabel 'N'\n");
			fprintf(w_no_dyn_sys, "set ylabel 'M'\n");

			fprintf(w_no_dyn_sys, "plot '%s' with image", no_dyn_sys_w_filename);

			fclose(w_no_dyn_sys);
		}
	}

}

//template prototypes
template void saveLearnCurveData<NO_DYN_SYS>(float* total_analysis_rewards);
template void saveLearnCurveData<DYN_SYS>(float* total_analysis_rewards);

template void saveErrorCurveData<NO_DYN_SYS>(float* total_reward_errors);
template void saveErrorCurveData<DYN_SYS>(float* total_reward_errors);

template void saveGraphAnalysisData<NO_DYN_SYS>(float* h_sumGrad, float* h_sumDeltaW, int step);
template void saveGraphAnalysisData<DYN_SYS>(float* h_sumGrad, float* h_sumDeltaW, int step);

template void saveWeightData<NO_DYN_SYS>(float* h_W, int step);
template void saveWeightData<DYN_SYS>(float* h_W, int step);

