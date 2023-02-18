clear all
close all
clc
rng('default');
X_test=importdata('test_data_3pars.txt');
X_test=X_test(:,3:end);
X_test=X_test';


range_Dgrey=[7.5230e-13 2.2569e-12];%m^2/s
range_Dwhite=5*range_Dgrey;
range_rho=[6.9445e-08 2.0833e-07]; %1/s

theta_test=importdata('glioma_3pars_test_set.txt');
theta_test=theta_test';

theta_test_scaled=[scaler(theta_test(:,1),range_Dgrey) ...
    scaler(theta_test(:,2),range_Dwhite)...
    scaler(theta_test(:,3),range_rho)];

X_train=importdata('train_data_3pars.txt');

X_train=X_train(:,3:end);
X_train=X_train';
N=size(X_train,1);

theta_train=importdata('glioma_3pars_train_set.txt');
theta_train=theta_train';

%look for NaN note that element 199 of every obs is a nan

[row, col] = find(isnan(X_train));
[row1,col1]= find(isnan(X_test));

X_train(:,col)=[];
X_test(:,col1)=[];


X_min=min(min(X_train));
X_max=max(max(X_train));
% rescaling training data in range [0,1]
X_train_scaled= (X_train-X_min)./(X_max-X_min);

%rescaling observed data in [0,1]
X_test_scaled= (X_test-X_min)./(X_max-X_min);

theta_train_scaled= [scaler(theta_train(:,1),range_Dgrey) ...
    scaler(theta_train(:,2),range_Dwhite)...
    scaler(theta_train(:,3),range_rho)];

[n_eval,n_points]=size(X_test_scaled);


%%
% Set up MDNetwork parameters.
nin = n_points;			% Number of inputs.
nhidden = 500;			% Number of hidden units.
ncentres = 2;			% Number of mixture components.
dim_target = 3;			% Dimension of target space
mdntype = '0';			% Currently unused: reserved for future use
alpha = 100;			% Inverse variance for weight initialisation
% Make variance small for good starting point
    
% Create and initialize network weight vector.
net = mdn(nin, nhidden, ncentres, dim_target, mdntype);
init_options = zeros(1, 18);
init_options(1) = -1;	% Suppress all messages
init_options(14) = 10;  % 10 iterations of K means in gmminit
net = mdninit(net, alpha, theta_train_scaled, init_options);
options = zeros(1,18);
options(1) = 1;			% This provides display of error values.
options(14) = 300;		% Number of training cycl
% Train using scaled conjugate gradients.
[net, options] = netopt(net, options,X_train_scaled,theta_train_scaled, 'scg');

loss = mdnerr(net, X_train_scaled,theta_train_scaled);
display(loss);

NPE_results=zeros(n_eval,9);

for i=1:n_eval
    x_eval=X_test_scaled(i,:);
    theta_eval=theta_test_scaled(i,:);

    %now we save and compare our estimated posterior for D, given u_
    out_mix = mdn2gmm(mdnfwd(net, X_test_scaled(i,:))); %computes the output of the MDN 
    % model in terms of mixture gaussian parameters
    
    mu_obs=out_mix.centres; %centroids
    prior_obs=out_mix.priors; %priors

    [~, ind] = max(prior_obs);
    error=sqrt(out_mix.covars(ind));
    
    predicted_obs_val=[inverse_normalizer(mu_obs(ind,1),range_Dgrey) ...
        inverse_normalizer(mu_obs(ind,2),range_Dwhite)...
        inverse_normalizer(mu_obs(ind,3),range_rho)];
    rescaled_error=[stan_dev_rescaler(error,range_Dgrey),...
        stan_dev_rescaler(error,range_Dwhite)...
        stan_dev_rescaler(error,range_rho)];
    
    NPE_results(i,1)=theta_test(i,1);
    NPE_results(i,2)=predicted_obs_val(1);
    NPE_results(i,3)=rescaled_error(1);

    NPE_results(i,4)=theta_test(i,2);
    NPE_results(i,5)=predicted_obs_val(2);
    NPE_results(i,6)=rescaled_error(2);

    NPE_results(i,7)=theta_test(i,3);
    NPE_results(i,8)=predicted_obs_val(3);
    NPE_results(i,9)=rescaled_error(3);

    %disp('Observed diffusion coefficient in grey matter')
    %disp(theta_test(i,1))
    %disp('Predicted diffusion coefficient in grey matter')
    %disp(predicted_obs_val(1))
    %disp('Interval for Dgrey coefficient')
    %disp([predicted_obs_val(1)-2*rescaled_error(1),predicted_obs_val(1)+2*rescaled_error(1)]);
%
    %disp('Observed diffusion coefficient in white matter')
    %disp(theta_test(i,2))
    %disp('Predicted diffusion coefficient in white matter')
    %disp(predicted_obs_val(2))
    %disp('Interval for Dwhite coefficient')
    %disp([predicted_obs_val(2)-2*rescaled_error(2),predicted_obs_val(2)+2*rescaled_error(2)]);
    %
    %disp('Observed growth rate')
    %disp(theta_test(i,3))
    %disp('Predicted growth rate')
    %disp(predicted_obs_val(3))
    %disp('Interval for rho coefficient')
    %disp([predicted_obs_val(3)-2*rescaled_error(3),predicted_obs_val(3)+2*rescaled_error(3)]);
end
%%
table_NPE=table(NPE_results(:,1),NPE_results(:,2),NPE_results(:,3),NPE_results(:,4),...
    NPE_results(:,5),NPE_results(:,6),NPE_results(:,7),NPE_results(:,8),NPE_results(:,9),...
'VariableNames',{'D_{grey} observed',...
    'D_{grey} predicted', 'standard error D_{grey}','D_{white} observed',...
    'D_{white} predicted', 'standard error D_{white}','\rho observed',...
    '\rho predicted','standard error \rho'});

disp(table_NPE)
