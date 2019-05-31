This is the readme file for the simulation code accompanying the paper
"Democratic population decisions result in robust policy-gradient
learning: a parametric study with GPU simulations" by Paul Richmond,
Lars Buesing, Michele Giugliano and Eleni Vasilaki, PLoS ONE
Neuroscience

To compile the code place the model directory into the CUDA SDK
 examples directory and compile using the visual studio 2008 project
 file provided. On a Linux environment it is possible to modify the
 CUDA SDKS 'template' example makefile to build the models CUDA files
 (*.cu).

The code has been written using a NVIDIA GTX480 (Fermi) GPU and as
such has been optimised and tested on this hardware using the visual
studio 2008 project and template files provided. In order to run the
simulations on older (non Fermi) hardware it may be necessary change
the THREADS_PER_BLOCK, SM_BLOCK_SIZE (model.cuh) and ind_configs
parameters (paramaters.h) to ensure the maximum number of thread
blocks does not exceed those supported by your hardware.

The following arguments can be used with the application executable:
	--profile : Profile the simulation only (i.e. no learning just analysis)
	--graph_analysis: Creates analysis graphs i.e. delta W and eligibility trace
	  (see paper). No learning takes place.
	--extended anlaysis: Same as -- graph_analysis however learning also takes 
	   place the graph analysis being performed before each learning step takes
	   place.
	--print_weight_plot: Prints the final weight values to a 3D plot after the 
	  simulation has run. Cannot be used with --graph_analysis as no learning 
	  takes place. If used with --extended_analysis then the weight plot is 
	  output after each learn step.
	--dyn_sys: only perform the simulation for the system with lateral 
	  connections
	--no_dyn_sys: only perform the simulation for the system with withou lateral
	  connections
	--device: usual CUDA device argument to specify the GPU device to use for 
	  simulation
  
