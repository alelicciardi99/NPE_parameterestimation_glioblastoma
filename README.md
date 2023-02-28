# NPE_parameterestimation_glioblastoma

This repository contains Matlab codes for the thesis of the exam of Mathematical Models for Biomedicine, a.y. 2022-23 at Politecnico di Torino, held by prof. Chiara Giverso, prof. Luigi Preziosi and prof. Luca Mesin. The work was made in cooperation with my colleagues Lorenzo Dal Zovo and Enrico Ortu. The MDN structures were implemented with the help of NetLab3.3 Matlab library.

The work aimed to develop a Neural Parameter Estimation (NPE) pipeline to predict diffusion and growth parameter of glioblastomas. All the data and packages needed to run and test this codes are in this repository.
Since this material had been developed to support a university course assignment, this repository provides all the steps and tests followed in order to obtain the definitive Machine Learning model for glioblastomas. Simulations and Data used to train and evaluate the model had been implemented using the software Comsol MultiPhyisics.

Authors: 
- Alessandro Licciardi s296152@studenti.polito.it
- Lorenzo Dal Zovo s304784@studenti.polito.it
- Enrico Ortu s305713@studenti.polito.it


 ## How to move through the folders:
  Here it is provided a short guide to present the contents of the folders.
  
  ### NPE_glioma_model
  
  This folder contains the definitive pipeline, the Comsol simulated dataset and all the results are summerized on printed tables, either for the two parameters model, either for the three parameters one.
  
  ### netlab-3.3
  
 Latest version of Netlab library (https://www.mathworks.com/matlabcentral/fileexchange/2654-netlab) - implemented by prof. Nabney, Ian. This library contains the codes for building Mixture Density Neural Nets models. With respect to the linked package, this folder is own-updated and modified in order to correctly run on Matlab23.
  
  ### rescaler_functions_matlab
  
  This folder groups some simple functions to rescale and standardize vectors and matrices, designed for the purpose of the NPE pipeline.
  
  ### NPE_model_development_matlab
  
  This folder summarizes all the early tests of MDNs and developments of the ML model that led to building the final pipeline. 
  
  ### comsol_trials_2pars
  
  This folder contains the first attempts and tests of the pipeline on the Comsol-based simulations of glioma growth with space-homogeneous diffusion and growth, thus the two parameters model.
  
  ### comsol_trials_2pars
  
  This folder contains the first attempts and tests of the pipeline on the Comsol-based simulations of glioma growth with non-homogeneous diffusion and growth, thus the three parameters model.
  
  
  
  
  
 
