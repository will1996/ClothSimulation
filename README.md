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






