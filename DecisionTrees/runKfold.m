function [trainEvaluation, testEvaluation] = runKfold(features, labels, treeType)

    % Split in 80% training and 20% testing 
    trainingSize = ceil(height(features)*0.8);
    trainingFeatures = features(1:trainingSize,:);
    trainingLabels = labels(1:trainingSize,:);
    testFeatures = features(trainingSize+1:height(features),:);
    testLabels = labels(trainingSize+1:height(features),:);

    trainEvaluation = cell(10,1); 
    testEvaluation = cell(10,1);
    
    % Split training data in 10 folds
    featureFolds = split(trainingFeatures);
    labelFolds = split(trainingLabels);
    
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
        testPredictions = runTree(testFeatures, decisionTree);
        
        if treeType == 0
            % Training data evaluation
            trainMetrics.rmse = calculateRMSE(validationSetLabels.label, trainingPredictions);
            % Testing data evaluation
            testMetrics.rmse = calculateRMSE(testLabels.label, testPredictions);
        elseif treeType == 1
            % Training data evaluation
            confusionMatrix = calculateConfusionMatrix(validationSetLabels.label, trainingPredictions);
            trainMetrics.recall = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FN);
            trainMetrics.precision = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FP);
            trainMetrics.F1Score = 2 * ( (trainMetrics.precision*trainMetrics.recall) / (trainMetrics.precision+trainMetrics.recall)  );
            % Testing data evaluation
            confusionMatrix = calculateConfusionMatrix(testLabels.label, testPredictions);
            testMetrics.recall = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FN);
            testMetrics.precision = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FP);
            testMetrics.F1Score = 2 * ( (testMetrics.precision*testMetrics.recall) / (testMetrics.precision+testMetrics.recall)  );
     
        end
        
        % Append results
        trainEvaluation{index} = trainMetrics;
        testEvaluation{index} = testMetrics;
        
    end
    
end

function splitTables = split(table)
    % Turn a table into an array of 10 sub-tables
    edges=[1:ceil(height(table)/10):height(table),height(table)];
    discretize(1:height(table), edges).';
    splitTables = splitapply(@(x) {cell2table(x, 'VariableNames', table.Properties.VariableNames)}, table2cell(table), discretize(1:height(table), edges).');
end