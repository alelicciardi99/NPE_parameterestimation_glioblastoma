clear all
close all
clc

%% Problem Set.
% The following script wants to learn the posterior pdf, adapting it
% to a given observation. Let us assume a simple model, e.g. 
% x(t)=e^(-t^2/sigma^2), which is a gaussian kernel. In our setting the
% probability distribution of the parameter sigma^2 is the goal of our
% analysis. We assume that log(p(sigma^2|x_obs)) is normal N(m,s^2). The
% technique is known as NPE, whiuch stands for neural parameter estimation.
% We start considering a prior p(theta) which is non-informative, and,
% in order to use normal priors, we are taking the logarithm of our
% sampled parameters.

%% noisy generation of obs_data
rng(10)

sd_err=0.025;
x=@(t,sigma2) exp(-t.^2./sigma2);
sigma2_obs=0.4;
t_1=0.5;
t_2=1.0;
u1_obs=x(t_1,sigma2_obs)+normrnd(0,sd_err);
u2_obs=x(t_2,sigma2_obs)+normrnd(0,sd_err);

figure(1)
title("observed data")
plot(linspace(0,2), x(linspace(0,2),sigma2_obs),'r');
hold on
plot([t_1,t_2],[u1_obs,u2_obs],'k*')
hold off
legend("theoretical evolution of obs", "available observed data ")
%%
prior_precision=1; %precision=1/variance
prior_mean=0;
N=100; %number of simulated samples
prior_sigma2=exp(normrnd(prior_mean,prior_precision^(-2),[N,1]));
u1_sim=x(t_1,prior_sigma2);
u2_sim=x(t_2,prior_sigma2);
%we plot a couple of prior simulated samples (trajectory and curve)
figure(2)
title('prior sampled data')
xline([t_1,t_2]);
hold on
for i=1:10
    plot(linspace(0,2),x(linspace(0,2),prior_sigma2(i)));
    hold on  
    plot([t_1,t_2],[u1_sim(i),u2_sim(i)],'k*')
end
hold off

%% RegNeuralNet training
% we are now interested in training a model to learn m and s2. 
% Our net should take as input the sampled u_1 and u_2 and then learn the
% posterior probability distribution, i.e. parameters m and s2. The
% following step is to validate according to our X_obs, if convergence
% criterion is not met, we sample from the new posterior, and this goes on
% until convergence is granted.
% Let us build the dataset.

X_data=[u1_sim,u2_sim]; %input are the two segments of image
y_out(:,1)=prior_mean*ones(N,1);
y_out(:,2)=-2*log(prior_precision)*ones(N,1);
%% 
model_NN_mean = fitrnet(X_data(:,1),y_out(:,1),...
    "LayerSizes",100, "Activations","sigmoid");
model_NN_var = fitrnet(X_data(:,1),y_out(:,2),...
   "LayerSizes",100, "Activations","sigmoid");
%%
mean_1=predict(model_NN_mean,u1_obs);
var_2=exp(predict(model_NN_var,u1_obs));
%% let us try to cycle and see if we improve our results

S=30;
pars=zeros(S,3);
for s=1:S
     prior_precision=1/var_2; %precision=1/variance
     prior_mean=mean_1;
     prior_sigma2=exp(normrnd(prior_mean,prior_precision^(-2),[N,1]));
     u1_sim=x(t_1,prior_sigma2);
     model_NN_mean = fitrnet(X_data(:,1),y_out(:,1),...
    "LayerSizes",100, "Activations","sigmoid");
     model_NN_var = fitrnet(X_data(:,1),y_out(:,2),...
    "LayerSizes",100, "Activations","sigmoid");

    
     y_out(:,1)=prior_mean*ones(N,1);
     y_out(:,2)=-2*log(prior_precision)*ones(N,1);
     mean_1=predict(model_NN_mean,u1_obs);
     var_2=exp(predict(model_NN_var,u1_obs));
     pars(s,1)=mean_1;
     pars(s,2)=exp(mean_1);
     pars(s,3)= var_2;




 end