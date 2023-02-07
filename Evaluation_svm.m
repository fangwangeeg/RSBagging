function[F1_score,accuracy_majority,accuracy_minority,accuracy_all,percent]= Evaluation_svm(results_all_basic_model, y_test, confidence_level)
% Evaluation_svm is a function to evaluate the performance of a RSBagging.
% The input includes results_all_basic_model, y_test, and confidence_level.
% The output includes F1_score, accuracy_majority, accuracy_minority, accuracy_all, and percent.

% Assign input results_all_basic_model to variable A
A = results_all_basic_model;
% Get the number of classifiers by subtracting 1 from the number of columns in A
Number_classifier = size(A,2)-1;

% For each row in A
for i=1:size(A,1)
    % Calculate the number of classifiers that predict class 1
    k(i,1)= length(find(A(i,2:Number_classifier+1)==1));
    % Calculate the number of classifiers that predict class 2
    k(i,2)= length(find(A(i,2:Number_classifier+1)==2));

    % If the number of classifiers that predict class 1 is greater than or equal to the confidence level times the number of classifiers
    if k(i,1)>=P_confidence*Number_classifier
        % Assign class 1 to the prediction for this row
        Predict_new(i,1) = 1;
        % Else if the number of classifiers that predict class 2 is greater than or equal to the confidence level times the number of classifiers
    elseif k(i,2)>=P_confidence*Number_classifier
        % Assign class 2 to the prediction for this row
        Predict_new(i,1) = 2;
        % Else, the prediction is not certain
    else
        Predict_new(i,1) = 0;
    end
end

% Find the indices of the rows in Predict_new where the prediction is not 0
index_predictable=find(Predict_new(:,1)~=0);
% Update y_test with the values from A corresponding to the predictable rows
y_test = A(index_predictable,1);
% Update H_new with the predictions corresponding to the predictable rows
H_new = Predict_new(index_predictable,1);
% Calculate the percentage of rows with a certain prediction
percent = length(index_predictable)/size(Predict_new,1);

% Initialize True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN) counts as 0
TP=0; FP=0; TN=0; FN=0;
% caculate confusion matrix
for j=1:length(y_test);
    if(H_new(j)==2 & y_test(j)==2);
        TP=TP+1;
    elseif(H_new(j)==2 & y_test(j)==1);
        FP=FP+1;
    elseif(H_new(j)==1 & y_test(j)==1);
        TN=TN+1;
    elseif(H_new(j)==1 & y_test(j)==2);
        FN=FN+1;
    end
end
% Caculate metrics for binary classification
accuracy_majority = 100*TN/(TN+FP);  % Specificity
accuracy_minority = 100*TP/(TP+FN);  % Sensivity
accuracy_all = 100*(TP+TN)/(FP+FN+TP+TN);
F1_score = (2*TP)/(2*TP+FP+FN);
end