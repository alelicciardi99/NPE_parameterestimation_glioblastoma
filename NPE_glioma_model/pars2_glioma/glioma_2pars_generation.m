clear all
close all
clc
rng('default');
N=1000;
D=rand(N,1);
rho=rand(N,1);
range_D=[7.5230e-13 2.2569e-12];%m^2/s
range_rho=[6.9445e-08 2.0833e-07]; %1/s
theta_target_un=[inverse_normalizer(D, range_D) inverse_normalizer(rho,range_rho)];

writematrix(theta_target_un','glioma_2pars','Delimiter',' ');