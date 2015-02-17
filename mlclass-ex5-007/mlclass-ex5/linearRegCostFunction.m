function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X = [12x2]
% y = [12x1]
% theta = [2x1]
h = X*theta; % [12x1]

% Compute J (without regularization)
J = 1/(2*m) * sum((h - y).^2);
% Regularize J (except theta0)
J = J + lambda/(2*m) * sum(theta(2:end).^2);

% Compute gradient (without regularization)
grad = 1/m * X' * (h - y); % [2x12] * [12x1] = [2x1]
% Regularize gradient (except theta0)
grad(2:end) = grad(2:end) + lambda/m*theta(2:end);

% =========================================================================

grad = grad(:);

end
