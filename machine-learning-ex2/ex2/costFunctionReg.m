function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
vars = length(theta);
regularization = 0;
for i = 2:vars
    regularization = regularization +theta(i)^2;
end
regularization = regularization * lambda / (2*m);

sig = sigmoid(X * theta);
J = ((y'*-1)*log(sig) - (1-y')*log(1-sig))/m + regularization;


grad = X'*(sig-y)/m;

for j = 2:vars
    grad(j,1) = grad(j,1)+lambda/m*theta(j);
end


% =============================================================

end
