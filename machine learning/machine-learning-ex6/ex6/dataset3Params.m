function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0; %1;
sigma = 0; %0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

[a,b] = meshgrid([0.01 0.03 0.1 0.3 1 3 10 30],[0.01 0.03 0.1 0.3 1 3 10 30]);
cartesian = [a(:) b(:) zeros(64,1)];

for i = 1:size(cartesian,1)
    model= svmTrain(X, y, cartesian(i,1), @(x1, x2) gaussianKernel(x1, x2, cartesian(i,2))); 
    predictions = svmPredict(model,Xval);
    cartesian(i,3) = mean(double(predictions ~= yval));
end

[minNum, minRow] = min(cartesian(:,3));
C = cartesian(minRow,1);
sigma = cartesian(minRow,2);

% =========================================================================

end
