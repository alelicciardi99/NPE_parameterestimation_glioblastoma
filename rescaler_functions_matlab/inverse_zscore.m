function A = inverse_zscore(Z,mu,sigma)
% Z is a standardized (by columns) matrix NxM
% mu is a 1XM vector containing the sample means of the cols of A
% sigma is a 1XM vector containing the sample standard deviations of the
% cols of A
A=Z.*sigma+mu;
end

