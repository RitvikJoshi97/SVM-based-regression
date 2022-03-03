clear; close all; clc;
% This code is using outer fold regression for linear Kernel Function
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
bestSVM_outerfold1(kouterFolds) = struct('SVMModel', NaN, ...     % this is to store the best SVM
        'C', NaN, 'Score', Inf,'Episilon',NaN);  

kouterIdx = crossvalind('Kfold', length(targets), kouterFolds);
    for k = 1:kouterFolds
        k
        kfolds=2;
        trainData = allData(kouterIdx~=k, :);
        trainTarg = targets(kouterIdx~=k);
        testData = allData(kouterIdx==k, :);
        testTarg = targets(kouterIdx==k);
        
        bestSVM_innerfold1 = struct('SVMModel', NaN, ...     % this is to store the best SVM
        'C', NaN, 'Score', Inf,'Episilon',NaN);

        bestSVM_innerfold1 = nested_cross_valid(kfolds,trainData,trainTarg)
        
        ["outerfolds: ",kouterFolds]
       

            bestSVM_outerfold1(k).SVMModel = bestSVM_innerfold1.SVMModel;
            bestSVM_outerfold1(k).C = bestSVM_innerfold1.C;
            bestSVM_outerfold1(k).Episilon = bestSVM_innerfold1.Episilon;
            bestSVM_outerfold1(k).Score = bestSVM_innerfold1.Score

    end

    [train,test] = crossvalind('HoldOut',length(Y),0.2);
   

for i=1:kouterFolds

    anSVMModel = fitrsvm(X(train), Y(train),'KernelFunction', 'linear', 'BoxConstraint', bestSVM_outerfold1(i).C,'Epsilon',bestSVM_outerfold1(i).Episilon);
    Y_hat=predict(anSVMModel,X(test));
    Y_hat_unscale=Y_hat*Sy_tab(1)+Cy_tab*ones(height(Y_hat(:,1)),1);
    test_MSE=(Y_hat_unscale-Y1(test)).'*(Y_hat_unscale-Y1(test))/length((Y_hat_unscale));
    sqrt(test_MSE)
end

function bestSVM =nested_cross_valid(kFolds,allData,targets)
    bestSVM = struct('SVMModel', NaN, ...     % this is to store the best SVM
        'C', NaN, 'Score', Inf,'Episilon',NaN);     
    
    kIdx = crossvalind('Kfold', length(targets), kFolds);
    for k = 1:kFolds
        k
        trainData = allData(kIdx~=k, :);
        trainTarg = targets(kIdx~=k);
        testData = allData(kIdx==k, :);
        testTarg = targets(kIdx==k);
          bestFeatCombo = struct('SVM', NaN, 'C', NaN);
          bestCScore = inf;
          bestScore = inf;
          bestC = NaN;
          bestRSVM = NaN;
          for epsilon = 0.1:0.1:1.2
                gridC = 2.^(-5:2:15);
                for C = gridC
                    anSVMModel = fitrsvm(trainData, trainTarg,'KernelFunction', 'linear', 'BoxConstraint', C,'Epsilon',epsilon);
                    L = loss(anSVMModel,testData, testTarg);
                    Y_hat=predict(anSVMModel,testData);
                    test_MSE=(Y_hat-testTarg).'*(Y_hat-testTarg)/length((Y_hat));
                     sqrt(test_MSE)
                    if test_MSE < bestCScore        % saving best SVM on parameter                  
                        bestCScore = test_MSE;      % selection
                        bestC = C;
                        bestCRSVM = anSVMModel;
                        bestCepsilon= epsilon;
                        
                    end
                end 
                
            if bestCScore < bestScore        % saving best SVM on parameter
                 bestScore = bestCScore;      % selection
                  bestC1 = bestC;
                  bestRSVM = bestCRSVM;
                  bestepsilon= bestCepsilon;
                  
            end  
          end
        % saving the best SVM over all folds
        if bestScore < bestSVM.Score
            bestSVM.SVMModel = bestRSVM;
            bestSVM.C = bestC1;
            bestSVM.Episilon = bestepsilon;
            bestSVM.Score = bestScore
        end
    end
end
