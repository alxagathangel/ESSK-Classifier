clc;clear;
datasets = {'glass','iris','lymphography','spambase','fertility'};
results = struct();

n = 20;
resultsMatrix = zeros(n, length(datasets));

for iter = 1:n
    fprintf('Iteration %d/%d\n', iter, n);

    for i = 1:length(datasets)
        datasetPath = fullfile('C:\[path to datasets]', datasets{i});
        
        dataFile = fullfile(datasetPath,[datasets{i},'.data']);
        labelColumn = 5;
        delimiter = ',';
        
        [data, labels] = loadDataset(dataFile, delimiter);
        
      % Train & test set SPLIT
        cv = cvpartition(size(data,1),'HoldOut',0.2);
        trainData = data(training(cv),:);
        trainLabels = labels(training(cv));
        testData = data(test(cv),:);
        testLabels = labels(test(cv));
        
      % ESSK Parameters
        K = 3; % neighbors
        m = 5; % iterations
        g = 1; % grace
        
        predictedLabels = ESSK(trainData,trainLabels,testData,K,m,g);
    
        accuracy = sum(predictedLabels == testLabels) / length(testLabels);
        results.(datasets{i}) = accuracy;
        resultsMatrix(iter, i) = accuracy;
    
    end
    
    disp('Classification Accuracies:');
    disp(results);

end

figure;
boxplot(resultsMatrix, datasets);
xlabel('Datasets');
ylabel('Accuracy');
title('Accuracy Distribution Across Iterations');


% Loading the dataset (assuming last column contains the label)
% INPUTS:
%   dataFile  - path to the dataset file
%   delimiter - divider of dataset
% OUTPUTS:
%   data   - all columns except the last
%   labels - last column
function [data, labels] = loadDataset(dataFile, delimiter)
    rawData = readtable(dataFile,'FileType','text','Delimiter',delimiter);

  % Separating features and labels
    numColumns = size(rawData, 2); 
    labels = rawData{:, numColumns};  % last column = labels
    data = rawData{:, 1:numColumns-1};  % rest of columns = features

  % For text columns...
    if iscell(labels) || isstring(labels) || iscategorical(labels)
        [labels, ~] = grp2idx(labels); % Convert text labels to numeric indices
    end

    data = normalize(data);
end

% Ensemble Sorted Subset KNN (ESSK)
% INPUTS:
%   trainData   - NxD matrix of training data
%   trainLabels - Nx1 vector of class labels corresponding to training data
%   testData    - MxD matrix of test data
%   K           - nearest neighbors
%   m           - iterations
%   g           - grace parameter (for extended search)
% OUTPUT:
%   predictedLabels - Mx1 vector of predicted class labels for test data.
function predictedLabels = ESSK(trainData, trainLabels, testData, K, m, g)

    [N, D] = size(trainData); % N=instances, D=features
    [~, M] = size(testData);  % M=test instances
    
  % Storing subsets & their sorted data
    SA = cell(m, 4);
    
    % TRAINING PHASE
    % -----------------------------------------------------------
    % Steps:
    %   1. Randomly select a subset of features
    %   2. Select best feature using Mutual Information (MI)
    %   3. Sort data based on best feature
    %   4. Store results
    % -----------------------------------------------------------
    for iter = 1:m   
    
    % 1:
        selectedFeatures = randperm(D, randi([1, D]));
    % 2:
        bestFeature = selectedFeatures(1);
        bestMI = -inf;
        for feature = selectedFeatures
            currentMI = mutualInformation(trainData(:, feature), trainLabels);
            if currentMI > bestMI
                bestMI = currentMI;
                bestFeature = feature;
            end
        end
    % 3:
        [sortedData, sortedIndices] = sortrows(trainData(:, bestFeature));
        sortedLabels = trainLabels(sortedIndices);
    % 4:    
        SA{iter, 1} = sortedData;
        SA{iter, 2} = sortedLabels;
        SA{iter, 3} = selectedFeatures;
        SA{iter, 4} = bestFeature;
    end
    
    % TESTING PHASE
    % -----------------------------------------------------------
    % Steps:
    %   1. Binary search to find approximate location
    %   2. Select neighbors within the grace window
    %   3. Perform KNN classification
    % -----------------------------------------------------------
    numTestInstances = size(testData, 1);
    predictions = zeros(numTestInstances, m);
    for t = 1:numTestInstances
        currentInstance = testData(t, :);
    
        for iter = 1:m
          % Subset details: 
            sortedData = SA{iter, 1};
            sortedLabels = SA{iter, 2};
            bestFeature = SA{iter, 4};
    % 1:
            low = 1;
            high = N;
            targetValue = currentInstance(bestFeature);
    
            while low <= high
                mid = floor(low + (high - low) / 2);
                if sortedData(mid) < targetValue
                    low = mid + 1;
                else
                    high = mid - 1;
                end
            end
    % 2: 
            left = max(1, low - K - g);
            right = min(N, low + K + g);
            neighborsData = trainData(left:right, :);
            neighborsLabels = sortedLabels(left:right);
    % 3:
            distances = sqrt(sum((neighborsData - currentInstance).^2, 2));
            [~, idx] = sort(distances);
            kNearestLabels = neighborsLabels(idx(1:K));
            predictions(t, iter) = mode(kNearestLabels);
        end
    end
    
    predictedLabels = mode(predictions, 2);

end

% Helper Function - Calculating mutual information between feature & labels
%   Uses histograms for discrete approximation.
% INPUTS:
%   feature - Nx1 vector of feature values
%   labels  - Nx1 vector of class labels
% OUTPUT:
%   MI - mutual information value
function MI = mutualInformation(feature, labels)

  % Map feature & labels to discrete integer values
    [~, ~, featureMapped] = unique(feature, 'rows');
    [~, ~, labelsMapped] = unique(labels, 'rows');

  % Joint & Marginal probabilities
    jointHist = accumarray([featureMapped, labelsMapped], 1);
    margFeature = sum(jointHist, 2); % marginal for features
    margLabels = sum(jointHist, 1); % marginal for labels
    total = sum(jointHist(:));

    jointProb = jointHist / total;
    margFeatureProb = margFeature / total;
    margLabelsProb = margLabels / total;

  % Calc MI
    MI = 0;
    [numFeatures, numLabels] = size(jointProb);
    for i = 1:numFeatures
        for j = 1:numLabels
            if jointProb(i, j) > 0
                MI = MI + jointProb(i,j) * log(jointProb(i,j) / (margFeatureProb(i) * margLabelsProb(j)));
            end
        end
    end
end