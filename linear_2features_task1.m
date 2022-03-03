
clear; close all; clc;
% This code is using Linear Kernel Function using only two features
%% preparing dataset
data_all=xlsread('C:\Users\psxrj4\Documents\MATLAB\winequality_for_classification.csv','A2:M3200');
labels = data_all(:,1:12);
red = data_all(:,13:13);

%% binary classification
rows = 3199;
X = randn(rows,10);
X(:,1:12) = labels(1:rows,:);
y = red(1:rows);

rand_num = randperm(size(X,1));
X_train = normalize(X(rand_num(1:round(0.8*length(rand_num))),:));
y_train = y(rand_num(1:round(0.8*length(rand_num))),:);

X_test = normalize(X(rand_num(round(0.8*length(rand_num))+1:end),:));
y_test = y(rand_num(round(0.8*length(rand_num))+1:end),:);

%% Best hyperparameter
f1 = 5;
f2 = 11;
x1 = X_train(:,f1:f1);
x2 = [x1,X_train(:,f2:f2)];
X_train_w_best_feature = x2;
Md1 = fitcsvm(X_train_w_best_feature,y_train,'KernelFunction','linear', 'BoxConstraint',1)


%% Final test with test set
x1_test = X_test(:,f1:f1);
x2_test = [x1_test,X_test(:,f2:f2)];
X_test_w_best_feature = x2_test;
y_pred = predict(Md1,X_test_w_best_feature);
test_accuracy_for_iter = sum(y_pred == y_test)/length(y_test)*100
%% plots 

figure;
gscatter(X_test_w_best_feature(:,1),X_test_w_best_feature(:,2),y_pred,'rb','xx');
hold on;
title('Plot of predicted Y against label Chlorides & label Alcohol')
h_sv=plot(Md1.SupportVectors(:,1),Md1.SupportVectors(:,2),'ko','markersize',8);
legend('red wine','white wine','support vectors');
xlabel('Chlordides');
ylabel('Alcohol');
hold off



SV_percent = sum(Md1.IsSupportVector==1)/length(Md1.IsSupportVector)*100


