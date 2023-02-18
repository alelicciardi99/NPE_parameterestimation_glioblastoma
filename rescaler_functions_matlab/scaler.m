function [scaled_A,max_A,min_A] = scaler(A,range)
%scales onto [0,1] over the columns of A
if nargin==1
       max_A=max(A);
       min_A=min(A);
else 
    max_A=range(2);
    min_A=range(1);
end

scaled_A=(A-min_A)./(max_A-min_A);
end

