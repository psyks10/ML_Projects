kIdx = getkFoldIdx(height(labels));

for k = 1:10

    trainIdx = horzcat(kIdx{setdiff(1:10,k)}).';
    trainData = features(trainIdx, :);
    trainLabels = labels(trainIdx,:);
    testData = features(kIdx{k}, :);
    testLabels = labels(kIdx{k},:);
    
    result = polynomialResults(k);


    model = fitcsvm(trainData, trainLabels, ...
            'KernelFunction', 'Polynomial', 'PolynomialOrder', result.q, ...
            'BoxConstraint', result.C);


    pTrainResults(k).C = result.C;
    pTrainResults(k).q = result.q;

    noOfSupportVectors =  height(model.SupportVectors);
    pTrainResults(k).NoOfSupportVectors = noOfSupportVectors;
    pTrainResults(k).PercentageOfSupportVectors = noOfSupportVectors/height(trainData);

    testPred = predict(model,testData);

    confusionMatrix = calculateConfusionMatrix(testLabels, testPred);
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
    pTrainResults(k).metrics = testMetrics;
    pTrainMetrics = [pTrainResults.metrics];

end

