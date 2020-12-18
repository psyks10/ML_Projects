
kIdx = getkFoldIdx(ceil(height(labels)));

for k = 1:10

    trainIdx = horzcat(kIdx{setdiff(1:10,k)}).';
    trainData = features(trainIdx, :);
    trainLabels = labels(trainIdx,:);
    testData = features(kIdx{k}, :);
    testTarg = labels(kIdx{k},:);
    
    result = guassianResults(k);

    if type
        model = fitcsvm(trainData, trainLabels, ...
            'KernelFunction', 'RBF', 'KernelScale', result.Gamma, ...
            'BoxConstraint', result.C);
    else 
        model = fitrsvm(trainData, trainLabels, ...
            'KernelFunction', 'RBF', 'KernelScale', result.Gamma, ...
            'BoxConstraint', result.C, 'Epsilon', result.Epsilon);
    end

    gTrainResults(k).C = result.C;
    gTrainResults(k).Gamma = result.Gamma;
    noOfSupportVectors =  height(model.SupportVectors);
    gTrainResults(k).NoOfSupportVectors = noOfSupportVectors;
    gTrainResults(k).PercentageOfSupportVectors = noOfSupportVectors/height(trainData);

    testPred = predict(model,testData);

    confusionMatrix = calculateConfusionMatrix(testTarg, testPred);
    testMetrics = struct('recall',NaN,'precision',NaN,'F1Score',NaN);
    if confusionMatrix.TP+confusionMatrix.FN==0
        testMetrics.recall = 0;
    else
        testMetrics.recall = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FN);  
    end
    
    if confusionMatrix.TP+confusionMatrix.FP==0
        testMetrics.precision = 0;
    else
        testMetrics.precision = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FP);
    end
    
    if testMetrics.recall==0
        testMetrics.F1Score = 0;
    else
        testMetrics.F1Score = 2 * ( (testMetrics.precision*testMetrics.recall) / (testMetrics.precision+testMetrics.recall)  );
    end
    
    gTrainResults(k).metrics = testMetrics;
    gTrainResults(k).labels = testPred.';
    gTrainMetrics = [gTrainResults.metrics];
end

