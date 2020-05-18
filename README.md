# Langevin_Machine_Learning

This repository contains codes to check various Langevin equations integrator, Lennard Jones Model with the visualization of the particles as well as Machine Learning Code for large time-step langevin integrator

Literature Review for the latest code can be found in MD ML 7.pdf file that explains the latest architecture, result and references.

_______________________________________________________________________________________________________________________________

# Documentation : 
  
  1 ) MD Lennard Jones contains all the code to visualize particles, initialization using periodic boundary condition (PBC),    with the main code at MD_final.py
 
 2 ) MD ML contains all the code in the past involving various model such as LSTM, RNN based, Convolutional Neural Network and correlation
  
  3 ) MD_SRNN contains the code for latest Hamiltonian Neural Network explained in the slides MD ML 7
  
  to use : 
  - Train the Model at MD_ML_Separate.py for 2 Separate Model as explained in the slides or General Hamiltonian for MD_ML.py
  - Get the distribution sampling from simulator_langevin.py
  - To get another temperature, generate using MSMC.py and then add the new temperature at simulator.py
