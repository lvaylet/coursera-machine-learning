function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0; % scalar
grad = zeros(size(theta)); % nx1 column vector

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% X = mxn matrix
% theta = nx1 column vector
h = sigmoid(X*theta); % h = mx1 column vector

% y = mx1 column vector
% log(h) = mx1 column vector
J = 1/m * sum(-y.*log(h) - (1-y).*log(1-h)); % J = scalar

% for example = 1:m
%     grad = grad + (h(example) - y(example)) * X(example,:)';
% end
% 
% grad = grad/m;

% Compute gradient (without regularization)
grad = 1/m * X' * (h - y);

% =============================================================

end
