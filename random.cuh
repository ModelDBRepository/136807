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
This CUDA header file provides the necessary device functions for an 
algorithm implementing random number generation on the GPU. The 
algorithm is based upon that described in the article "Harvesting 
graphics power for MD simulations" by J.A. van Meel, A. Arnold, 
D. Frenkel, S. F. Portegies Zwart and R. G. Belleman, Molecular Simulation, 
Vol. 34, p. 259 (2007) distributed under the GNU GPL v2.
*/

#ifndef _RANDOM_CUDA_HEADER_
#define _RANDOM_CUDA_HEADER_


__device__ static uint2 RNG_rand48_iterate_single(rand48seeds Xn, magic_numbers mn)
{
	unsigned int R0, R1;

	// low 24-bit multiplication
	const unsigned int lo00 = __umul24(Xn.x, mn.x);
	const unsigned int hi00 = __umulhi(Xn.x, mn.x);

	// 24bit distribution of 32bit multiplication results
	R0 = (lo00 & 0xFFFFFF);
	R1 = (lo00 >> 24) | (hi00 << 8);

	R0 += mn.z; R1 += mn.w;

	// transfer overflows
	R1 += (R0 >> 24);
	R0 &= 0xFFFFFF;

	// cross-terms, low/hi 24-bit multiplication
	R1 += __umul24(Xn.y, mn.x);
	R1 += __umul24(Xn.x, mn.y);

	R1 &= 0xFFFFFF;

	return make_uint2(R0, R1);
}


__device__ float rand48(rand48seeds* seeds,magic_numbers mn, int width)
{
	
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	index += blockIdx.y*width;

	uint2 state = seeds[index];

	int rand = ( state.x >> 17 ) | ( state.y << 7);

	// this actually iterates the RNG
	state = RNG_rand48_iterate_single(state, mn);

	seeds[index] = state;

	return (float)rand/2147483647;
}

#endif