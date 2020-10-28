
/// Software requirements

You need to install FEniCS version 2017.2.0 to run this code, which can be cloned as:

""

FENICS_VERSION='2017.2.0'
git clone --branch=$FENICS_VERSION https://bitbucket.org/fenics-project/dolfin
git clone --branch=$FENICS_VERSION https://bitbucket.org/fenics-project/mshr

""

The instructions for installing the respective libraries are provided in their bitbucket repositories.

Important Remark: As 2017.2.0 is not the latest version of the library, you need to apply the patch 'fenics-2017.2.0-parse_doxygen.patch' before the build as follows:

""

patch dolfin/doc/parse_doxygen.py -i /path_to_the_patch/fenics-2017.2.0-parse_doxygen.patch

""


/// Steps in running the code

1) You can generate the meshes in hdf5 format using the script 'run_bisection_mesh_generator_2d.py' for the different test cases with straightforward modifications. The meshes generated thus are saved in the folder "meshes". Some examples of the meshes used in the test cases are provided already.

2) The convergence studies have been set-up in the following scripts:

- run_test1_type1.py
- run_test1_type2.py
- run_test2_type1.py
- run_test2_type2.py

Here, type1 refers to the case when polynomial degree of 'v' and 'sigma' are the same, and type2 refers to the case when polynomial degree of 'sigma' is 1 less than that of the polynomial degree of 'v'.

The output files generated from these scripts are saved in the folder "output/waveO1".

3) Plotting scripts corresponding to the test cases are provided in the folder "scripts". You can generate different plots with straightforward modifications in these scripts.


/// Important folders:

1) mesh_generator: Contains the routines for bisection refinement and defines the polygon geometries.

2) test_cases:  Contains the different test problems for the first-order wave equation.

3) systems: Contains the DG spatial discretisation for the wave equation.

4) src: Contains DG time-discretisation, referred here as 'time-integrator', error computation and sparse-grids handlers.
