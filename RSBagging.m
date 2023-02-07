function [MeanSD_result,result] = classification_ensemble_classifiers(confidence_level,label,feature,num_base_classifiers, average_run_times,test_rng)
y = label;
X = feature;
number_classifier = num_base_classifiers;
number_average_model = average_run_times;
for jj=1:number_average_model
for i=1:number_classifier 
%% 
%split data to balanced train set and imblance test set
[X_train,y_train,X_test,y_test] = split_undersampling_imbalance_test(X,y,test_rng)
end
%% normalize dataset
[train_feature, mean_x, std_x] = normalize_data(X_train, [], []);
test_feature = normalize_data(X_test, mean_x, std_x);
X_train=train_feature;
X_test=test_feature;
% process NAN
X_train_= rmmissing(X_train');
X_train = X_train_';
X_test_= rmmissing(X_test');
X_test = X_test_';
%% bisic classifiers
classification = fitcsvm(...
    X_train, ...
    y_train, ...
    'KernelFunction', 'rbf', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [1; 2]);

[H,probability] = predict(classification,X_test);
 % hard voting
 A(:,1)= y_test;
 A(:,i+1)=H;
end
[F1_score,accuracy_awake,accuracy_disorder,accuracy_all,percent]= Evaluation_svm(A,confidence_level)
result(jj,1)= accuracy_all;
result(jj,2)= accuracy_awake;
result(jj,3)= accuracy_disorder;
result(jj,4)= F1_score;
result(jj,5)= percent;
 MeanSD_result(1:5)=nanmean(result(:,:),1);
 MeanSD_result(6:10)=nanstd(result(:,:),0);
end

