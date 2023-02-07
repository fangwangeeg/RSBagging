function [X, mean_X, std_X] = normalize_data(X, mean_X, std_X)
if isempty(mean_X) && isempty(std_X)
    mean_X = mean(X);
    std_X = std(X);
    X = (X - repmat(mean_X, size(X,1), 1)) ./ repmat(std_X, size(X,1), 1);
else
    X = (X - repmat(mean_X, size(X,1), 1)) ./ repmat(std_X, size(X,1), 1);
end
end