clear; close all; clc;
% This code is using nested cross validation with outer fold regression for RBF Kernel Function
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
        'C', NaN, 'Score', Inf,'Sigma',NaN,'Episilon',NaN);  

kouterIdx = crossvalind('Kfold', length(targets), kouterFolds);
    for k = 1:kouterFolds
        k
        kfolds=2;
        trainData = allData(kouterIdx~=k, :);
        trainTarg = targets(kouterIdx~=k);
        testData = allData(kouterIdx==k, :);
        testTarg = targets(kouterIdx==k);
        
        bestSVM_innerfold1 = struct('SVMModel', NaN, ...     % this is to store the best SVM
        'C', NaN, 'Score', Inf,'Sigma',NaN,'Episilon',NaN);

        bestSVM_innerfold1 = nested_cross_valid(kfolds,trainData,trainTarg)
        
        ["outerfolds: ",kouterFolds]
       

        bestSVM_outerfold1(k).SVMModel = bestSVM_innerfold1.SVMModel;
        bestSVM_outerfold1(k).C = bestSVM_innerfold1.C;
        bestSVM_outerfold1(k).sigma=bestSVM_innerfold1.Sigma;
        bestSVM_outerfold1(k).Episilon = bestSVM_innerfold1.Episilon;
        bestSVM_outerfold1(k).Score = bestSVM_innerfold1.Score
    end

    [train,test] = crossvalind('HoldOut',length(Y),0.2);
   

for i=1:kouterFolds

    anSVMModel = fitrsvm(X(train), Y(train),'KernelFunction', 'RBF','KernelScale',bestSVM_outerfold1(i).sigma, 'BoxConstraint', bestSVM_outerfold1(i).C,'Epsilon',bestSVM_outerfold1(i).Episilon);
    Y_hat=predict(anSVMModel,X(test));
    Y_hat_unscale=Y_hat*Sy_tab(1)+Cy_tab*ones(height(Y_hat(:,1)),1);
    test_MSE=(Y_hat_unscale-Y1(test)).'*(Y_hat_unscale-Y1(test))/length((Y_hat_unscale));
    sqrt(test_MSE)
end

function bestSVM =nested_cross_valid(kFolds,allData,targets)
    bestSVM = struct('SVMModel', NaN, ...     % this is to store the best SVM
        'C', NaN, 'Score', Inf,'Sigma',NaN,'Episilon',NaN);     
    
    kIdx = crossvalind('Kfold', length(targets), kFolds);
    for k = 1:kFolds
        k
        trainData = allData(kIdx~=k, :);
        trainTarg = targets(kIdx~=k);
        testData = allData(kIdx==k, :);
        testTarg = targets(kIdx==k);
          bestFeatCombo = struct('SVM', NaN, 'C', NaN);
          bestCScore = inf;
          bestCSigmaScore=inf;
          bestScore = inf;
          bestC = NaN;
          bestRSVM = NaN;
          for epsilon = 0.1:0.1:1.2
              for sigma = 0.1:0.1:2.0
                gridC = 2.^(-5:2:15);
                for C = gridC
                    anSVMModel = fitrsvm(trainData, trainTarg,'KernelFunction', 'RBF', 'KernelScale', sigma,'BoxConstraint', C,'Epsilon',epsilon);
                    L = loss(anSVMModel,testData, testTarg);
                    Y_hat=predict(anSVMModel,testData);
                    test_MSE=(Y_hat-testTarg).'*(Y_hat-testTarg)/length((Y_hat));
                    sqrt(test_MSE)
                    if test_MSE < bestCScore        % saving best SVM on parameter                  
                        bestCScore = test_MSE;      % selection
                        bestC = C;
                        bestCsigma = sigma;
                        bestCRSVM = anSVMModel;
                        bestCepsilon= epsilon;
                        
                    end
                end 
                if bestCScore < bestCSigmaScore        % saving best SVM on parameter
                        
                        bestCSigmaScore = bestCScore;      % selection
                        bestSigmaC = bestC;
                        bestsigma1 = bestCsigma;
                        bestSigmaRSVM = bestCRSVM;
                        bestSigmaepsilon= bestCepsilon;
                end
              end
            if bestCSigmaScore < bestScore        % saving best SVM on parameter
                 bestScore = bestCSigmaScore;      % selection
                  bestC1 = bestSigmaC;
                  bestsigma2 = bestsigma1;
                  bestRSVM = bestSigmaRSVM;
                  bestepsilon= bestSigmaepsilon;
                  
            end  
          end
        % saving the best SVM over all folds
        if bestScore < bestSVM.Score
            bestSVM.SVMModel = bestRSVM;
            bestSVM.C = bestC1;
            bestSVM.Sigma=bestsigma2;
            bestSVM.Episilon = bestepsilon;
            bestSVM.Score = bestScore
        end
    end
end
