clear all
close all
clc
rng('default');
u_obs=importdata('data_obs.txt');
XY=u_obs(:,1:2);
u_obs=u_obs(:,3)';

range_D=[7.5230e-13 2.2569e-12];%m^2/s
range_rho=[6.9445e-08 2.0833e-07]; %1/s

theta_obs=[0.25 0.65];
theta_obs_un=[inverse_normalizer(theta_obs(1),range_D) inverse_normalizer(theta_obs(2),range_rho)];

X_data=importdata('results_0_it.txt');

X_data=X_data(:,3:end);
X_data=X_data';
N=size(X_data,1);

theta_target=importdata('par_sweep_0.txt');
theta_target=theta_target';

%look for NaN note that element 199 of every obs is a nan

[row, col] = find(isnan(X_data));
[row1,col1]= find(isnan(u_obs));

X_data(:,col)=[];
u_obs(:,col1)=[];


X_min=min(min(X_data));
X_max=max(max(X_data));
% rescaling training data in range [0,1]
X_data_scaled= (X_data-X_min)./(X_max-X_min);

%rescaling observed data in [0,1]
X_obs_scaled= (u_obs-X_min)./(X_max-X_min);

%
%range_D= [min(theta_target(:,1)) max(theta_target(:,1))];
%range_rho= [min(theta_target(:,2)) max(theta_target(:,2))];

%
theta_norm= scaler(theta_target);
theta_obs(1)= scaler(theta_obs_un(1),range_D);
theta_obs(2)= scaler(theta_obs_un(2),range_rho);
n_points=length(X_obs_scaled);
%%
f=figure();
p1=histogram(theta_norm(:,1),10);
hold on
p2=xline(theta_obs(1),'r_','LineWidth',2);
hold off
%%
  
% Set up MDNetwork parameters.
nin = n_points;			% Number of inputs.
nhidden = 500;			% Number of hidden units.
ncentres = 2;			% Number of mixture components.
dim_target = 2;			% Dimension of target space
mdntype = '0';			% Currently unused: reserved for future use
alpha = 100;			% Inverse variance for weight initialisation
% Make variance small for good starting point

% Create and initialize network weight vector.
net = mdn(nin, nhidden, ncentres, dim_target, mdntype);
init_options = zeros(1, 18);
init_options(1) = -1;	% Suppress all messages
init_options(14) = 10;  % 10 iterations of K means in gmminit
net = mdninit(net, alpha, theta_norm, init_options);
options = zeros(1,18);
options(1) = 1;			% This provides display of error values.
options(14) = 300;		% Number of training cycl
% Train using scaled conjugate gradients.
[net, options] = netopt(net, options,X_data_scaled,theta_norm, 'scg');

loss = mdnerr(net, X_data_scaled,theta_norm);
display(loss);

%now we save and compare our estimated posterior for D, given u_
out_mix = mdn2gmm(mdnfwd(net, X_obs_scaled)); %computes the output of the MDN 
% model in terms of mixture gaussian parameters

mu_obs=out_mix.centres; %centroids
prior_obs=out_mix.priors; %priors
sigma_obs=cat(3,diag(out_mix.covars(1)*ones(2,1)),diag(out_mix.covars(2)*ones(2,1))); %variances
gm_obs = gmdistribution(mu_obs,sigma_obs,prior_obs); %build a gaussian 
% mixture object
gmPDF_obs = @(x,y) arrayfun( @(x0,y0) pdf(gm_obs,[x0,y0]), x,y); %saving the pdf 


f1=figure;
fsurf(gmPDF_obs,[-2,2])
hold on
plot3(theta_obs(1)*ones(100,1),theta_obs(2)*ones(100,1),linspace(0,7),'r','LineWidth',2)
hold on
plot3(theta_obs(1),theta_obs(2),0,'rx','LineWidth',2)

theta_target = random(gm_obs,N);

%remove samples not in [0,1] range
theta_target(theta_target(:,1)<0,:)=[];
theta_target(theta_target(:,2)<0,:)=[];
theta_target(theta_target(:,1)>1,:)=[];
theta_target(theta_target(:,2)>1,:)=[];
N_size=size(theta_target,1);



theta_target_un=[inverse_normalizer(theta_target(:,1),range_D)...
inverse_normalizer(theta_target(:,2),range_rho)];

ind=1;
ind2=2;
if prior_obs(2)>prior_obs(1)
    ind=2;
    ind2=1;
end
other_var=out_mix.covars(ind2);
error=(out_mix.covars(ind));

predicted_obs_val=[inverse_normalizer(mu_obs(ind,1),range_D) inverse_normalizer(mu_obs(ind,2),range_rho)];
rescaled_error=[stan_dev_rescaler(error,range_D),stan_dev_rescaler(error,range_rho)];

disp('Observed diffusion coefficient')
disp(theta_obs_un(1))
disp('Predicted diffusion coefficient')
disp(predicted_obs_val(1))
disp('Interval for D coefficient')
disp([predicted_obs_val(1)-2*rescaled_error(1),predicted_obs_val(1)+2*rescaled_error(1)]);
disp('Observed growth rate')
disp(theta_obs_un(2))
disp('Predicted growth rate')
disp(predicted_obs_val(2))
disp('Interval for rho coefficient')
disp([predicted_obs_val(2)-2*rescaled_error(2),predicted_obs_val(2)+2*rescaled_error(2)]);


%writematrix(theta_target_un','out_pars1');
