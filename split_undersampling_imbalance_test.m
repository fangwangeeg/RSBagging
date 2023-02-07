function [X_train,y_train,X_test,y_test] = split_undersampling_imbalance_test(X,y,test_rng)

% the ratio of test set
ratio = 0.2;

% find the majority class and minority class in the labels 
maj = mode(y); % the majority class
min = setdiff(unique(y), maj); % the minority class

% separate the majority and minority class data
y_maj = y(y == maj);
y_min = y(y == min);
X_maj = X(y == maj, :);
X_min = X(y == min, :);

% test set
y_test_min_index = randperm(numel(y_min), floor(numel(y_min)*ratio));
y_test_min = y_min(y_test_min_index);
X_test_min = X_min(y_test_min_index, :);
y_test_maj_index = randperm(numel(y_maj), floor(numel(y_maj)*ratio));
y_test_maj = y_maj(y_test_maj_index);
X_test_maj = X_maj(y_test_maj_index, :);

% train set with minority class
y_train_min_index = setdiff(1:numel(y_min), y_test_min_index);
y_train_min = y_maj(y_train_min_index);
X_train_min = X_maj(y_train_min_index, :);

% train set with majority class
y_train_maj_index = datasample(setdiff(1:numel(y_maj), y_test_maj_index), numel(y_train_min),'Replace', false );
y_train_maj = y_maj(y_train_maj_index);
X_train_maj = X_maj(y_train_maj_index, :);

% combine the two classes 
X_train = [X_train_maj; X_train_min];
y_train = [y_train_maj; y_train_min];
X_test = [X_test_maj; X_test_min];
y_test = [y_test_maj; y_test_min];
end