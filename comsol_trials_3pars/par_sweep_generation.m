clear all
close all
clc
rng('default');
N=200;
Dgrey=rand(N,1);
Dwhite=rand(N,1);
rho=rand(N,1);
range_Dgrey=[7.5230e-13 2.2569e-12];%m^2/s
range_Dwhite=5*range_Dgrey;
range_rho=[6.9445e-08 2.0833e-07]; %1/s
theta_target_un=[inverse_normalizer(Dgrey, range_Dgrey)...
    inverse_normalizer(Dwhite, range_Dwhite)...
    inverse_normalizer(rho,range_rho)];

writematrix(theta_target_un','par_sweep_0_3d','Delimiter',' ');