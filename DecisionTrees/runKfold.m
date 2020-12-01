function accuracy = runKfold(features, labels)

    featureFolds = split(features);
    labelFolds = split(labels);

    accuracy = 0;
    
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
        
        validationSetFeatures = featureFolds{validationSplit};
        validationSetLabels = labelFolds{validationSplit};
        
        decisionTree = decisionTreeLearning(trainingSetFeatures, trainingSetLabels);
        runTree(validationSetFeatures, decisionTree);
        
        DrawDecisionTree(decisionTree, 'Classification Tree');

    end
    
end

function splitTables = split(table)
    edges=[1:ceil(height(table)/10):height(table),height(table)];
    discretize(1:height(table), edges).';
    splitTables = splitapply(@(x) {cell2table(x, 'VariableNames', table.Properties.VariableNames)}, table2cell(table), discretize(1:height(table), edges).');
end