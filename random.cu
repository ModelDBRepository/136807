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
This CUDA source file provides the necessary host functions for initialising 
an algorithm implementing random number generation on the GPU. The 
algorithm is based upon that described in the article "Harvesting 
graphics power for MD simulations" by J.A. van Meel, A. Arnold, 
D. Frenkel, S. F. Portegies Zwart and R. G. Belleman, Molecular Simulation, 
Vol. 34, p. 259 (2007) distributed under the GNU GPL v2.
*/

#include <cutil_inline.h>
#include "random.h"


/* Host code init function */
void initCUDARand48(unsigned int max_rand, rand48seeds* h_seeds, rand48seeds* d_seeds, magic_numbers &mn)
{
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	srand ( (unsigned int) time(NULL) );
	int seed = rand();
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	
	for (unsigned int i = 0; i < max_rand; ++i) {
		C += A*c;
		A *= a;
	}
	
	//magic numbers
	mn.x = A & 0xFFFFFFLL;
	mn.y = (A >> 24) & 0xFFFFFFLL;
	mn.z = C & 0xFFFFFFLL;
	mn.w = (C >> 24) & 0xFFFFFFLL;

	//prepare MAX_RAND numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i=0; i<max_rand; i++)
	{
		x = a*x + c;
		h_seeds[i].x = x & 0xFFFFFFLL;
		h_seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}

	//copy seeds to device
	CUDA_SAFE_CALL( cudaMemcpy( d_seeds, h_seeds, max_rand*sizeof(rand48seeds), cudaMemcpyHostToDevice));

}



