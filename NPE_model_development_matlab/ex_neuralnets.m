clear all
close all
clc
rng(10)
sigma_err=1e-1;
N=40;
f=@(x) 5*x.*exp(-x.^2);

x_data=linspace(-2,2,N)';
y_data=f(x_data)+normrnd(0,sigma_err,[N,1]);

test_size=0.3;
data=[x_data,y_data];
%%
% Cross varidation (train: 70%, test: 30%)
cv = cvpartition(size(data,1),'Holdout',test_size);
idx = cv.test;

% Separate to training and test data
dataTrain = data(~idx,:);
dataTest  = data(idx,:);

%%
figure(1)
plot(x_data,y_data,'r')
hold on
plot(dataTrain(:,1),dataTrain(:,2),'r*')
hold on
plot(dataTest(:,1),dataTest(:,2),'k*')
hold off

%% training the NNet

model_NN = fitrnet(dataTrain(:,1),dataTrain(:,2),...
    "LayerSizes",10, "Activations","sigmoid")
%% 
testMSE = loss(model_NN,dataTest(:,1),dataTest(:,2))
%%
testPredictions = predict(model_NN,dataTest(:,1));
plot(dataTest(:,1),testPredictions,".")
hold on
plot(dataTest(:,1),dataTest(:,2),"ko")
hold off
%% testing on the whole data
DataValid=linspace(-2,2,100)';
YValid=f(DataValid)+normrnd(0,sigma_err,[100,1]);
testMSE = loss(model_NN,DataValid,YValid)
validPredictions = predict(model_NN,DataValid);
figure(3)
plot(DataValid,validPredictions,".")
hold on
plot(DataValid,YValid,"ko")
hold on
plot(DataValid,f(DataValid),'r')
hold off
legend('model prediction','ground truth', 'denoised function')


