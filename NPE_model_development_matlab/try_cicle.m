%% we want to avoid numbers not in [0,1] by sampling from a pdf 
% A is normal(1,0.9);
A=normrnd(1,1,[20,2])
A(A(:,1)<0,:)=[];
A(A(:,2)<0,:)=[];
A





