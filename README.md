# ClothSimulation

Highly paralell cloth simulation using a hybrid temporal- spatial paralellization method. 



Build instructions for UMD servers:

Environment setup: 
Requires Scons, python 2.7, cuda 10.1.243, openmpi4.0.1 gcc 7.5.0
it probably works with other combinations, but these are the most tested so far. 

Use  the module system to get cuda, openmpi and gcc with: 


module load cuda/10.1.243
module load openmpi/4.0.1
module load gcc/7.5.0

Next, make sure your anaconda environment is set up with python 2.7 and scons. 


To Build: 

make -C dependencies 
make 

# Steps to modify main branch to junbangcode branch
Download Miniconda3
https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Create environment using .yml file https://gist.github.com/will1996/6a19ffd36997486ab049268b3444db3d/archive/de7ebc2f4235a3904fb89567d979284df435b0a6.zip

Remove any lines from .yml that CONDA is unable to immediately solve. Let CONDA resolve them for you

Edit dependencies/Makefile to replace /* with /linux in taucs rules

(TODO) Compile Taucs so that it sees the cilk.h file

Run make in dependencies/ directory

Add -g flag to top level make file for debugging.

Add /lib64 to link /usr/lib/libgcc_s.so.1 to top level Make file

Run make in top level director to compile arcsim. See https://github.com/DanielTakeshi/ARCSim-Installation-Instructions for more details

`./bin/arcsim simulateoffline conf1/<conf>.json out`

Ensure <conf>.json file references a mesh file that exists within meshes/

Add .asString() to if-statement comparison for json[“disable”], conf.cpp:128

Increase max_iters in collision.cpp

Hardcode string instead of using dynamic `outprefix`, dosimulation.cpp:89

Run `mpirun -n <num proc> bin/arcsim simulateoffline conf1/<conf>.json out




