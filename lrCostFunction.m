function [J, Grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
Grad = zeros(size(theta));

% Each row of the resulting matrix will contain the value of the
% prediction for that example. You can make use of this to vectorize
% the cost function and gradient computations. 

theta(1)=0;

H= sigmoid(X * theta);

J = -mean(y.*log(H) + (1 - y).*log(1 - H))+ (lambda/2).*mean(theta(2:end).^2);

% =============================================================

Grad(1,1) = (1/m)*(X(:,1)'*(H-y)); 
Grad(2:end,1)=((1/m)*(X(:,2:end))'*(H-y))+(lambda/m)*theta(2:end);

Grad = Grad(:);

end
