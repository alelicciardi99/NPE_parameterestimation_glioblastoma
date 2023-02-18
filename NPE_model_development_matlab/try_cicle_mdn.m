%% Problem Setting
% The following example is set to simply introduce and test the 
% procedure for parameter estimation with MDNs, i.e. mixture density
% neural nets.
% The idea is the following. We have a function u(t;D), where D is 
% the targeted parameter.
% We are working at a fixed time, say t*, and we have availability
% of data u_obs= u(t*;D_obs). Our goal is to iteratively learn the
% posterior probability distribution 
% p(D|u_obs)= sum_k{alpha_k^obs f(D;mu_k^obs,sigma_k^obs)}, where
% f(D;mu,sigma) is a gaussian pdf. 
% The idea is to start generating a set of
% data points {(u_i,D_i),i=1,...,n} where D_i are sampled from an
% uninformative prior p(D) and u_i=u(t*;D_i). We then train a MDN
% taking u_i as covariate, and D_i as target variable and we give an
% estimate of alpha_k(u),mu_k(u),sigma_k(u). We then feed our MDN with
% u_obs, and learn the posterior parameters alpha_k(u_obs),mu_k(u_obs),
% sigma_k(u_obs). Then simulate other D_i from the posterior p(D|u_obs) and
% repeat until convergence is reached.
% Furthermore one should also decide a stopping criterion, which may
% involve a check if the sigma_k^obs at s-iteration corresponding to the
% largest alpha_k^obs is small enough.
% For this example we choose u(t;D)=exp(-tD), and we work with a bi-modal
% gaussian mixture.

%% Generate uninformative data

clear all 
close all
clc

N=1000; %number of simulated data parameters
t_star=0.5; %we fix our time t*
u=@(D) exp(-D*t_star);
rng('default') % for reproducibility
D_obs=0.8; %we fix our target parameter, which is the target of our estimate
u_obs=u(D_obs)+normrnd(0,0.05); %our observation is perturbed by some random 
%error, which in reality may be caused by measurment errors.

%we suppose that our initial prior distribution is a uniform distribution 
% over (0,20), as we suppose that for such phoenomena the parameter should 
%be in this range for any individual

D=20*rand(N,1);

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
p2=plot(u_obs,D_obs,'r*');
hold off
legend([p1,p2],'training uninformative data', 'targeted observation')
title('scatter plot observed data vs parameters')
xlabel('measured value of u')
ylabel('D parameter')
pause

%% Cycle for convergence

% In this example we assume a bimodal gaussian mixture for our parameter.
% Since this example is a beginners' example, we are not developping yet a
% fine stopping criterion, we fix the number of iteration and see how the
% model evolves.
 
max_iter=5; %maximum iteration number
s=1; % iteration counter
posterior_pars=zeros(max_iter,6); %to save posterior pars
for s=1:max_iter
    % Set up MDNetwork parameters.
    nin = 1;			% Number of inputs.
    nhidden = 100;			% Number of hidden units.
    ncentres = 2;			% Number of mixture components.
    dim_target = 1;			% Dimension of target space
    mdntype = '0';			% Currently unused: reserved for future use
    alpha = 100;			% Inverse variance for weight initialisation
    % Make variance small for good starting point
    
    % Create and initialize network weight vector.
    net = mdn(nin, nhidden, ncentres, dim_target, mdntype);
    init_options = zeros(1, 18);
    init_options(1) = -1;	% Suppress all messages
    init_options(14) = 10;
    % 10 iterations of K means in gmminit
    net = mdninit(net, alpha, D, init_options);

    options = zeros(1,18);
    options(1) = 1;			% This provides display of error values.
    options(14) = 200;		% Number of training cycles.

    % Train using scaled conjugate gradients.
    [net, options] = netopt(net, options,X_data,D, 'scg');

    plotvals = linspace(min(X_data),max(X_data))'; %value range in which D is expected 
    % to be found

    mixes = mdn2gmm(mdnfwd(net, plotvals)); %learn the parameter of 
    % the mixture, recall that they are functions of the input of the MDN

    y = zeros(1, length(plotvals));
    priors = zeros(length(plotvals), ncentres);
    c = zeros(length(plotvals), 2);
    widths = zeros(length(plotvals), ncentres);
    for i = 1:length(plotvals)
      [m, j] = max(mixes(i).priors);
      y(i) = mixes(i).centres(j,:);
      c(i,:) = mixes(i).centres';
    end

    fh1=figure;
    p1 = plot(X_data, D, '--y');
    hold on
    p2 = plot(plotvals, y, '*r');
    legend([p1 p2], 'training data', 'MDN mode max alpha centroids');
    title('learning performance on training data')
    xlabel('u values data')
    ylabel('D parameter values')
    hold off

   

    %now we show the mixture parameters functions

    fh3 = figure;

    subplot(3, 1, 1)
    plot(plotvals, c)
    hold on
    title('Mixture centres')
    legend('centre 1', 'centre 2')
    hold off
    
    priors = reshape([mixes.priors], mixes(1).ncentres, size(mixes, 2))';
   
    subplot(3, 1, 2)
    plot(plotvals, priors)
    hold on
    title('Mixture priors')
    legend('centre 1', 'centre 2')
    hold off

    variances = reshape([mixes.covars], mixes(1).ncentres, size(mixes, 2))';
    %%fh4 = figure;
    subplot(3, 1, 3)
    plot(plotvals, variances)
    hold on
    title('Mixture variances')
    legend('centre 1', 'centre 2')
    hold off
    
    
    
    %now we save and compare our estimated posterior for D, given u_obs

    out_mix = mdn2gmm(mdnfwd(net, u_obs)); %computes the output of the MDN 
    % model in terms of mixture gaussian parameters
    
    mu_obs=out_mix.centres; %centroids
    prior_obs=out_mix.priors; %priors
    sigma_obs=cat(3,out_mix.covars(1),out_mix.covars(2)); %variances
    
    gm_obs = gmdistribution(mu_obs,sigma_obs,prior_obs); %build a gaussian 
    % mixture object
    
    gmPDF_obs = @(x) arrayfun( @(x0) pdf(gm_obs,x0), x); %saving the pdf as
    %array fun
    
    fh4=figure;
    D_vals=linspace(0.5,2.5)';
    p1=plot(linspace(0.5,2.5),gmPDF_obs(linspace(0.5,2.5)),'LineWidth',2);
    hold on
    p2=xline(D_obs);
    title('posterior pdf of D given u_obs')
    legend([p1 p2],'posterior pdf','parameter value')
    hold off

    pause
    posterior_pars(s,:)=[prior_obs,mu_obs',out_mix.covars(1),out_mix.covars(2)];
    %generate new data from the posterior
    
    close(fh3);
    close(fh1);
    close(fh4);
    
    D = random(gm_obs,N);
    X_data=u(D)+normrnd(0,0.1,[N,1]);
    


end
             

