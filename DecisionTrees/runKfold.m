function accuracy = runKfold(features, labels, treeType)

    featureFolds = split(features);
    labelFolds = split(labels);

    accuracy = cell(10,1);
    
    possibleSplits = calculatePossibleSplits(1:10);

    for index = 1:10

        validationSplit = possibleSplits{index,1};
        trainingSplit = possibleSplits{index,2};
        
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
        
        decisionTree = decisionTreeLearning(trainingSetFeatures, trainingSetLabels, treeType);
        predictions = runTree(validationSetFeatures, decisionTree);
        
        if treeType == 0
            metrics.rmse = calculateRMSE(validationSetLabels.label, predictions);
        elseif treeType == 1
            confusionMatrix = calculateConfusionMatrix(validationSetLabels.label, predictions);
            metrics.recall = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FN);
            metrics.precision = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FP);
            metrics.F1Score = 2 * ( (metrics.precision*metrics.recall) / (metrics.precision+metrics.recall)  );
        end
        
        accuracy{index} = metrics;
        
    end
    
end

function splitTables = split(table)
    edges=[1:ceil(height(table)/10):height(table),height(table)];
    discretize(1:height(table), edges).';
    splitTables = splitapply(@(x) {cell2table(x, 'VariableNames', table.Properties.VariableNames)}, table2cell(table), discretize(1:height(table), edges).');
end