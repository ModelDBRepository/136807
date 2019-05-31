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
This CUDA header file contains the template function prototypes used for 
creating data plots.
*/

#ifndef _BANDIT_OUTPUT_H
#define _BANDIT_OUTPUT_H

#include "model.h"
#include "parameters.cuh"


template<DYNAMICS dynamics> void saveLearnCurveData(float* total_analysis_rewards);
void createLearnCurveGraph(bool dyn_sys, bool no_dyn_sys);

template<DYNAMICS dynamics> void saveErrorCurveData(float* total_reward_errors);
void createErrorCurveGraph(bool dyn_sys, bool no_dyn_sys);

template<DYNAMICS dynamics> void saveGraphAnalysisData(float* h_sumGrad, float* h_sumDeltaW, int step=0);
void createAnalysisGraphs(bool dyn_sys, bool no_dyn_sys, int step=0);


template<DYNAMICS dynamics> void saveWeightData(float* h_W, int step=0);
void createWeightGraphs(bool dyn_sys, bool no_dyn_sys, int step=0);


#endif _BANDIT_OUTPUT_H
