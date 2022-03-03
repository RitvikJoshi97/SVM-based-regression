clear; close all; clc;
% This code is using nested cross validation with outer fold regression for Polynomial Kernel Function
M=readtable('CCPP.xlsx');
N=normalize(M);
features=4;
X1=table2array(M(:,1:features));
Y1=table2array(M(:,features+1));
[N_tab,C_tab,S_tab]= normalize(X1,'zscore');
[Ny_tab,Cy_tab,Sy_tab]= normalize(Y1,'zscore');
X=N_tab;
Y=Ny_tab;

allData = X(1:200,1:features);
targets = Y(1:200,1);
length(targets)

kouterFolds = 10;
bestSVM_outerfold(kouterFolds) = struct('SVMModel', NaN, ...     % this is to store the best SVM
        'C', NaN, 'Score', Inf,'Sigma',NaN,'Episilon',NaN); 

kouterIdx = crossvalind('Kfold', length(targets), kouterFolds);
    for k = 1:kouterFolds
        k
        kfolds=2;
        trainData = allData(kouterIdx~=k, :);
        trainTarg = targets(kouterIdx~=k);
        testData = allData(kouterIdx==k, :);
        testTarg = targets(kouterIdx==k);


        bestSVM_innerfold = struct('SVMModel', NaN, ...     % this is to store the best SVM in innerfold
        'C', NaN, 'Score', Inf,'q',NaN,'Episilon',NaN);   

        bestSVM_innerfold =nested_cross_valid(kfolds,trainData,trainTarg)
        
        ["outerfolds: ",kouterFolds]

        bestSVM_outerfold(k).SVMModel = bestSVM_innerfold.SVMModel;
        bestSVM_outerfold(k).C = bestSVM_innerfold.C;
        bestSVM_outerfold(k).q=bestSVM_innerfold.q;
        bestSVM_outerfold(k).Episilon = bestSVM_innerfold.Episilon;
        bestSVM_outerfold(k).Score = bestSVM_innerfold.Score
    end

    [train,test] = crossvalind('HoldOut',length(Y),0.2);
   

for i=1:kouterFolds

    anSVMModel = fitrsvm(X(train), Y(train),'KernelFunction', 'polynomial','PolynomialOrder',bestSVM_outerfold(i).q, 'BoxConstraint', bestSVM_outerfold(i).C,'Epsilon',bestSVM_outerfold(i).Episilon);
    Y_hat=predict(anSVMModel,X(test));
    Y_hat_unscale=Y_hat*Sy_tab(1)+Cy_tab*ones(height(Y_hat(:,1)),1);
    test_MSE=(Y_hat_unscale-Y1(test)).'*(Y_hat_unscale-Y1(test))/length((Y_hat_unscale));
    sqrt(test_MSE)
end
   
function bestSVM =nested_cross_valid(kFolds,allData,targets)
        %  number of folds
    bestSVM = struct('SVMModel', NaN, ...     % this is to store the best SVM
        'C', NaN, 'Score', Inf,'q',NaN,'Episilon',NaN);     
    
    kIdx = crossvalind('Kfold', length(targets), kFolds);
    for k = 1:kFolds  %inner cross validation
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
                gridC = 2.^(-5:2:9);
                for C = gridC
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
end

