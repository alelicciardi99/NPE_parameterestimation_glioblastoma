function rescaled_error=stan_dev_rescaler(input_sdev,range)
%this function rescales the error in the parameters domain
%input_sdev is the standard deviation in [0,1]
% range=[a,b] is the parameter domain
% since X in [0,1] is mapped in [a,b] via the linear transformation
% Y=(b-a)*X+a, Var(Y)=(b-a)^2*Var(X):
rescaled_error=(range(2)-range(1))*input_sdev;
end