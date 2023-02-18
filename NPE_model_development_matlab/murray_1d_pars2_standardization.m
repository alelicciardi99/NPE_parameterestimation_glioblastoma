%% Problem Set:
% In this script we are interested in finding out if normalizing (via the z-score) 
% the training data at each iteration leads to a better performance.
% For such task we keep using Murray's model on brain tumour growth with
% constant diffusion D and constant net growth rate rho.

clear all
close all
clc

t_star=4;
N=1000;
n_points=100;
rng('default');
D_obs=8e-4;
rho_obs=0.15;
theta_obs=[D_obs,rho_obs];

%simulate data
range_rho=[0.075 0.575];
range_D=[4.8e-4 48e-4];
rho_target=inverse_normalizer(rand(N,1),range_rho);
D_target=inverse_normalizer(rand(N,1),range_D);
theta_target=[D_target,rho_target];
%u=@(x,D) exp(-x.^2/(D*t_star)); with this kernel works pretty well

u=@(x,theta) 10./(4*pi*theta(1)*t_star).*exp(theta(2)*t_star-x.^2/(4*theta(1)*t_star));

%Murray in-vitro tumour growth
x_points=linspace(-2,2,n_points);

X_obs=u(x_points,theta_obs)+normrnd(0,0.01,[1,n_points]);

X_data=zeros(N,n_points);

for i=1:N
    X_data(i,:)=u(x_points,theta_target(i,:))+normrnd(0,0.01,[1,n_points]);
end


figure(1)
for i=1:8
    plot(x_points,X_data(i,:),'LineWidth',2);
    hold on
end
hold on
plot(x_points,X_obs,'r--','LineWidth',2)
hold off

tol=00;

s=1;
loss=1e5;
[theta_std,means_L,s_devs_L]=zscore(theta_target);
theta_obs_std=(theta_obs-means_L)./s_devs_L;
%%
while (loss>tol)

    %standardize the data
    
    
    % Set up MDNetwork parameters.
    nin = n_points;			% Number of inputs.
    nhidden = 350;			% Number of hidden units.
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
    net = mdninit(net, alpha, theta_std, init_options);

    options = zeros(1,18);
    options(1) = 1;			% This provides display of error values.
    options(14) = 200;		% Number of training cycles.

    % Train using scaled conjugate gradients.
    [net, options] = netopt(net, options,X_data,theta_std, 'scg');

    loss = mdnerr(net, X_data,theta_std);
    display(loss);
    
    %now we save and compare our estimated posterior for D, given u_obs

    out_mix = mdn2gmm(mdnfwd(net, X_obs)); %computes the output of the MDN 
    % model in terms of mixture gaussian parameters
    
    mu_obs=out_mix.centres; %centroids
    prior_obs=out_mix.priors; %priors
    sigma_obs=cat(3,diag(out_mix.covars(1)*ones(2,1)),diag(out_mix.covars(2)*ones(2,1))); %variances
    gm_obs = gmdistribution(mu_obs,sigma_obs,prior_obs); %build a gaussian 
    % mixture object
    gmPDF_obs = @(x,y) arrayfun( @(x0,y0) pdf(gm_obs,[x0,y0]), x,y); %saving the pdf as
   
    
    theta_sampled= random(gm_obs,N);
    %[theta_std,means,s_devs]=zscore(theta_target);
    theta_target=inverse_zscore(theta_std,means_L,s_devs_L);
  
    for i=1:N
        X_data(i,:)=u(x_points,theta_target(i,:))+normrnd(0,0.01,[1,n_points]);
    end

      f1=figure;
    fsurf(gmPDF_obs,[-2,2])
    hold on
    plot3(theta_obs_std(1)*ones(100,1),theta_obs_std(2)*ones(100,1),linspace(0,7),'r','LineWidth',2)
    
    pause
    close(f1)
   
s=s+1;
end
ind=1;
if prior_obs(2)>prior_obs(1)
    ind=2;
end
predicted_vals=inverse_zscore(mu_obs(ind,:),means_L,s_devs_L);
%predicted_obs_val=[inverse_normalizer(predicted_normalized(1),range_D) inverse_normalizer(predicted_normalized(2),range_rho)];
display('Observed values:');
display(theta_obs);

display('Predicted values');
display(predicted_vals);


