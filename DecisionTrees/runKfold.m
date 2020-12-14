function [trainEvaluation] = runKfold(features, labels, treeType)

    trainEvaluation = cell(10,2); 
    
    % Split training data in 10 folds
    featureFolds = split(features);
    labelFolds = split(labels);
    
    % Calculate all possible combinations for the 10 folds
    possibleSplits = calculatePossibleSplits(1:10);

    for index = 1:10
        % Get the fold numbers corresponding to training folds and
        % validation fold for this iteration
        validationSplit = possibleSplits{index,1};
        trainingSplit = possibleSplits{index,2};
        
        % Separate data into training and validation
        trainingSetFeatures = table();
        trainingSetLabels = table();
        for i=1:length(trainingSplit)
            trainingSetFeatures = vertcat(trainingSetFeatures,featureFolds{i});
            trainingSetLabels = vertcat(trainingSetLabels,labelFolds{i});    
        end
        
        trainingSetLabels.label = num2cell(trainingSetLabels.label);
        
        validationSetFeatures = featureFolds{validationSplit};
        validationSetLabels = labelFolds{validationSplit};
        validationSetLabels.label = num2cell(validationSetLabels.label);
        
        % Generate a decision tree and its predictions on training and test
        % data
        decisionTree = decisionTreeLearning(trainingSetFeatures, trainingSetLabels, treeType);
        trainingPredictions = runTree(validationSetFeatures, decisionTree);
        
        if treeType == 0
            % Training data evaluation
            trainMetrics.rmse = calculateRMSE(validationSetLabels.label, trainingPredictions);
        elseif treeType == 1
            % Training data evaluation
            confusionMatrix = calculateConfusionMatrix(validationSetLabels.label, trainingPredictions);
            trainMetrics.recall = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FN);
            trainMetrics.precision = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FP);
            trainMetrics.F1Score = 2 * ( (trainMetrics.precision*trainMetrics.recall) / (trainMetrics.precision+trainMetrics.recall)  );
        end
        trainMetrics.labels = trainingPredictions;

        % Append results
        trainEvaluation{index,1} = trainMetrics;        
    end
    
end

function splitTables = split(table)
    % Turn a table into an array of 10 sub-tables
    edges=[1:ceil(height(table)/10):height(table),height(table)];
    discretize(1:height(table), edges).';
    splitTables = splitapply(@(x) {cell2table(x, 'VariableNames', table.Properties.VariableNames)}, table2cell(table), discretize(1:height(table), edges).');
end