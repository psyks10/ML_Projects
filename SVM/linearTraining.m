function results = linearTraining(type, features, labels)

    if type
        results = struct('NoOfSupportVectors', NaN, 'Metrics', NaN, 'labels', NaN);
    else
        results = struct('Epsilon', NaN, 'RMSE', NaN,'NoOfSupportVectors', NaN, 'labels', NaN);
    end
    
    trainIdx = 1:ceil(height(features)*0.8);
    testIdx =  setdiff(1:height(features),trainIdx);

    trainData = features(trainIdx, :);
    trainLabels = labels(trainIdx,:);

    testData = features(testIdx, :);
    testLabels = labels(testIdx,:);
    
    if type
        epsilons = 1:1;
    else
        epsilons = {0.1,0.2,0.5,1};
    end
    
    for i = 1:length(epsilons)
        if type
            model = fitcsvm(trainData,trainLabels, 'KernelFunction','linear', 'BoxConstraint',1);
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
            results(i).Metrics = testMetrics;
        else
            model = fitrsvm(trainData,trainLabels, 'KernelFunction','linear', 'BoxConstraint',1,'Epsilon', epsilons{i});
            testPred = predict(model,testData);
            results(i).Epsilon = epsilons{i};
            results(i).RMSE = calculateRMSE(testLabels, testPred);
        end
        
        noOfSupportVectors =  height(model.SupportVectors);
        results(i).NoOfSupportVectors = noOfSupportVectors;
        results(i).PercentageOfSupportVectors = noOfSupportVectors/height(trainData);
        results(i).labels = testPred.';
    end
    
    
end