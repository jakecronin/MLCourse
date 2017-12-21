function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%Cost is summation of all rows m of -(y/m)log(theta) 

% evaluate data points with theta values: X * theta
% Evaluate with sigmoid to map to y: (-1,1) range: sig=sigmoid(X*theta)
% calculate cost with log (convex evaluation):
%   -y*log(sig)-((1-y)*log(1-sig))
sig = sigmoid(X * theta); %h(X)

J = ((-1*y')*log(sig)-((-1*y'+1)*log(1-sig))) / m;

grad = X' * (sig-y) ./ m;

% =============================================================

end
