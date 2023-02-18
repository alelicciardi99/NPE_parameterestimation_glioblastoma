clear all
close all
clc

%%
t_star=1.6;
N=1000;
rng('default');
D_obs=3.6;
D_target=20*rand(N,1)+0.1;
u=@(x,D) exp(-x.^2/(D*t_star));
n_points=100;
x_points=linspace(-2,2,n_points);
X_obs=u(x_points,D_obs)+normrnd(0,0.01,[1,n_points]);
X_data=zeros(N,n_points);
figure(1)
histogram(D_target)
for i=1:N
    X_data(i,:)=u(x_points,D_target(i))+normrnd(0,0.01,[1,n_points]);
end
figure(2)
for i=1:8
    plot(x_points,X_data(i,:),'LineWidth',2);
    hold on
end
hold on
plot(x_points,X_obs,'r--','LineWidth',2)
hold off
%%
max_iter=4; %maximum iteration number

posterior_pars=zeros(max_iter,6); %to save posterior pars
for s=1:max_iter
    % Set up MDNetwork parameters.
    nin = n_points;			% Number of inputs.
    nhidden = 150;			% Number of hidden units.
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
    net = mdninit(net, alpha, D_target, init_options);

    options = zeros(1,18);
    options(1) = 1;			% This provides display of error values.
    options(14) = 200;		% Number of training cycles.

    % Train using scaled conjugate gradients.
    [net, options] = netopt(net, options,X_data,D_target, 'scg');

    %plotvals = linspace(min(X_data),max(X_data))'; %value range in which D is expected 
    % to be found

    %mixes = mdn2gmm(mdnfwd(net, plotvals)); %learn the parameter of 
    % the mixture, recall that they are functions of the input of the MDN

  %y = zeros(1, length(plotvals));
  %priors = zeros(length(plotvals), ncentres);
  %c = zeros(length(plotvals), 2);
  %widths = zeros(length(plotvals), ncentres);
  %for i = 1:length(plotvals)
  %  [m, j] = max(mixes(i).priors);
  %  y(i) = mixes(i).centres(j,:);
  %  c(i,:) = mixes(i).centres';
  %end
%
   %fh1=figure;
   %p1 = plot(X_data, D, '--y');
   %hold on
   %p2 = plot(plotvals, y, '*r');
   %legend([p1 p2], 'training data', 'MDN mode max alpha centroids');
   %title('learning performance on training data')
   %xlabel('u values data')
   %ylabel('D parameter values')
   %hold off

   

    %now we show the mixture parameters functions

    %fh3 = figure;
%
    %subplot(3, 1, 1)
    %plot(plotvals, c)
    %hold on
    %title('Mixture centres')
    %legend('centre 1', 'centre 2')
    %hold off
    
   % priors = reshape([mixes.priors], mixes(1).ncentres, size(mixes, 2))';
   
   %subplot(3, 1, 2)
   %plot(plotvals, priors)
   %hold on
   %title('Mixture priors')
   %legend('centre 1', 'centre 2')
   %hold off

    %variances = reshape([mixes.covars], mixes(1).ncentres, size(mixes, 2))';
    %%fh4 = figure;
    %subplot(3, 1, 3)
    %plot(plotvals, variances)
    %hold on
    %title('Mixture variances')
    %legend('centre 1', 'centre 2')
    %hold off
    
    
    
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
    D_vals=linspace(0.5,10)';
    p1=plot(D_vals,gmPDF_obs(D_vals),'LineWidth',2);
    hold on
    p2=xline(D_obs);
    title('posterior pdf of D given u_obs')
    legend([p1 p2],'posterior pdf','parameter value')
    hold off

    pause
    posterior_pars(s,:)=[prior_obs,mu_obs',out_mix.covars(1),out_mix.covars(2)];
    %generate new data from the posterior
    
    %close(fh3);
    %close(fh1);
    close(fh4);
    
    D_target = random(gm_obs,N);
    for i=1:N
        X_data(i,:)=u(x_points,D_target(i))+normrnd(0,0.01,[1,n_points]);
    end

end
             

