clear; close all; clc;
% This code is using nested cross validation with inner fold regression for Polynomial Kernel Function
M=readtable('CCPP.xlsx');
features=4;
allData = normalize(table2array(M(1:200,1:features)));
targets = normalize(table2array(M(1:200,features+1)));
length(targets)
featSize = size(allData, 2);
kFolds = 10;     %  number of folds
bestSVM = struct('SVMModel', NaN, ...     % this is to store the best SVM
    'C', NaN, 'Score', Inf,'q',NaN,'Episilon',NaN);     

kIdx = crossvalind('Kfold', length(targets), kFolds);
for k = 1:kFolds
    trainData = allData(kIdx~=k, :);
    trainTarg = targets(kIdx~=k);
    testData = allData(kIdx==k, :);
    testTarg = targets(kIdx==k);
      bestFeatCombo = struct('SVM', NaN, 'C', NaN);
      bestCScore = inf;
      bestCqScore=inf;
      bestScore = inf;
      bestC = NaN;
      bestRSVM = NaN;
      for epsilon = 0.1:0.1:1.2
          for q = 2:1:5
            gridC = 2.^(-5:2:15);
            for C = gridC
                C
                 [q,C,epsilon,k]
                anSVMModel = fitrsvm(trainData, trainTarg,'KernelFunction', 'polynomial','PolynomialOrder',q, 'BoxConstraint', C,'Epsilon',epsilon);
                L = loss(anSVMModel,testData, testTarg);
                Y_hat=predict(anSVMModel,testData);
                test_MSE=(Y_hat-testTarg).'*(Y_hat-testTarg)/length((Y_hat));
                sqrt(test_MSE)
                if test_MSE < bestCScore        % saving best SVM on parameter                  
                    bestCScore = test_MSE;      % selection
                    bestC = C;
                    bestCq = q;
                    bestCRSVM = anSVMModel;
                    bestCepsilon= epsilon;
                    
                end
            end 
            if bestCScore < bestCqScore        % saving best SVM on parameter
                    
                    bestCqScore = bestCScore;      % selection
                    bestqC = bestC;
                    bestq1 = bestCq;
                    bestqRSVM = bestCRSVM;
                    bestqepsilon= bestCepsilon;
            end
          end
        if bestCqScore < bestScore        % saving best SVM on parameter
             bestScore = bestCqScore;      % selection
              bestC1 = bestqC;
              bestq2 = bestq1;
              bestRSVM = bestqRSVM;
              bestepsilon= bestqepsilon;
              
        end  
      end
    % saving the best SVM over all folds
    if bestScore < bestSVM.Score
        bestSVM.SVMModel = bestRSVM;
        bestSVM.C = bestC1;
        bestSVM.q=bestq2;
        bestSVM.Episilon = bestepsilon;
        bestSVM.Score = bestScore
    end
end
SV_percent=sum(bestSVM.SVMModel.IsSupportVector==1)/length(bestSVM.SVMModel.IsSupportVector)*100