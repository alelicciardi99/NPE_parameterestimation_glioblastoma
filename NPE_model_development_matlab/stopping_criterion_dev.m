%% Script description:
% In this script we aim to define a stopping criterion, i.e. we want to
% find when our predicted posterior is accurate enough - recall that in
% real world experiment we do not know what is the value of the observed
% parameter. A simple idea may be to set a desired tolerance on the loss
% function value of the MDNet, e.g. a small value such as 1e-14 might get
% the job done.
%%

clear all
close all
clc

%% Let us focus on the case of a one parameter function, spatially homogeneous
t_star=1.5;
u = @(D) (1+exp(-D.*t_star)).^(-1); %logistic-like growth

N=1000; %number of simulated data parameters

rng('default') % for reproducibility
D_obs=5; %we fix our target parameter, which is the target of our estimate
X_obs=u(D_obs)+normrnd(0,0.05); %our observation is perturbed by some random 
%error, which in reality may be caused by measurment errors.

D=10*rand(N,1);

figure(1)
histogram(D)
legend('sampled parameters')
title('histogram of sampled values for D')



X_data=u(D)+normrnd(0,0.05,[N,1]); %for the sake of simplicity we perturb with the
%same noise as u_obs, without loss of generality this may occur after
%statistical analysis on measurment tools.

figure(2)
p1=plot(X_data,D,'bo');
hold on
p2=plot(X_obs,D_obs,'r*');
hold off
legend([p1,p2],'training uninformative data', 'targeted observation')
title('scatter plot observed data vs parameters')
xlabel('measured value of u')
ylabel('D parameter')

tol=0;
stop_crit=false;
s=1; %iteration counter

while (stop_crit==false)

    % Set up MDNetwork parameters.
    nin = 1;			% Number of inputs.
    nhidden = 50;			% Number of hidden units.
    ncentres = 2;			% Number of mixture components.
    dim_target = 1;			% Dimension of target space
    mdntype = '0';			% Currently unused: reserved for future use
    alpha = 100;			% Inverse variance for weight initialisation
    % Make variance small for good starting point
    
    % Create and initialize network weight vector.
    net = mdn(nin, nhidden, ncentres, dim_target, mdntype);
    init_options = zeros(1, 18);
    init_options(1) = -1;	% Suppress all messages
    init_options(14) = 10;  % 10 iterations of K means in gmminit
    net = mdninit(net, alpha, D, init_options);

    options = zeros(1,18);
    options(1) = 1;			% This provides display of error values.
    options(14) = 250;		% Number of training cycles.

    % Train using scaled conjugate gradients.
    [net, options] = netopt(net, options,X_data,D, 'scg');

     loss = mdnerr(net, X_data, D);
     display(loss);
     
     if (loss <= tol)
         stop_crit=true;
     end

      %now we save and compare our estimated posterior for D, given u_obs

    out_mix = mdn2gmm(mdnfwd(net, X_obs)); %computes the output of the MDN 
    % model in terms of mixture gaussian parameters
    
    mu_obs=out_mix.centres; %centroids
    prior_obs=out_mix.priors; %priors
    sigma_obs=cat(3,out_mix.covars(1),out_mix.covars(2)); %variances

    gm_obs = gmdistribution(mu_obs,sigma_obs,prior_obs); %build a gaussian 
    % mixture object
    
    gmPDF_obs = @(x) arrayfun( @(x0) pdf(gm_obs,x0), x); %saving the pdf as
    %array fun

    fh4=figure;
    D_vals=linspace(0,12)';
    p1=plot(D_vals,gmPDF_obs(D_vals),'LineWidth',2);
    hold on
    p2=xline(D_obs);
    title('posterior pdf of D given u_obs')
    legend([p1 p2],'posterior pdf','parameter value')
    hold off

    pause
   
    %generate new data from the posterior 
    close(fh4);
    D = random(gm_obs,N);
    X_data=u(D)+normrnd(0,0.05,[N,1]);
    s=s+1;


end




