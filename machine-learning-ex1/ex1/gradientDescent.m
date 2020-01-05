function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	%theta = theta-(alpha*(1/m*( theta'.*X - y).*X))
	%cost = computeCost(X, y, theta)
	%c = sprintf('%d ', cost);
	%fprintf('thetaNew 1 =  %s\n', c);
	%der = ((1/m)*(X*theta - y)'*X)'
	%theta = theta - alpha*der;
	
	
	x = X(:,2); % get the last column
    h = theta(1) + (theta(2)*x); % calculate the hypothesis 

    theta_zero = theta(1) - alpha * (1/m) * sum(h-y); % update the new theta zero
    theta_one  = theta(2) - alpha * (1/m) * sum((h - y) .* x); % update the new theta one

    theta = [theta_zero; theta_one]; % update theta
    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

	%fprintf('thetaNew 1 =  %d\n', theta);
	%fprintf('size Theta =  %d\n', size(theta));

end

end
