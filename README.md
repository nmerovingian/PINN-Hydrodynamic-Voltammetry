# PINN-Hydrodynamic-Voltammetry
 Solving convective-diffusion mass transport problem for channel electrode

This is a code repository in company with "The Application of Physics-Informed Neural Networks to Hydrodynamic Voltammetry" submitted to *Analyst*

# Requirements
Python 3.7 and above is suggested to run the program. The neural networks was developed and tested with Tensorflow 2.3. To install required packages, run

```
$ pip install -r requirement.txt

```

# Content
This repository has four folders for the four cases illustrated in the paper. They are:
* PINN-2D Channel 4 micron: Simulation of channel electrode assuming electrode length of 4 micron to predict steady state current 
* PINN-2D Channel 11 micron: Simulation of channel electrode assuming electrode length of 11 micron to predict steady state current 
* PINN-2D Double Channel: Simulation of double channel electrode to obtain collection efficiency
* PINN-2D CE channel: Simulation of channel electrode with CE reaction to obtain kinetic current


# Issue Reports
Please report any issues/bugs of the code in the discussion forum of the repository or contact the corresponding author of the paper


# Cite
To cite, please refer to [Link](https://doi.org/10.1039/D2AN00456A)