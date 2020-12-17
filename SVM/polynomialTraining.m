function results = polynomialTraining(type, features, labels)    

    if type
        results = struct('C', NaN, 'NoOfSupportVectors', NaN, 'labels', NaN);
    else
        results = struct('C', NaN, 'Epsilon', NaN, 'RMSE', NaN,'NoOfSupportVectors', NaN, 'labels', Na);
    end
    
    kIdx = getkFoldIdx(height(labels));

    for k = 1:10
      
        trainIdx = horzcat(kIdx{setdiff(1:10,k)}).';
        trainData = features(trainIdx, :);
        trainLabels = labels(trainIdx,:);
        testData = features(kIdx{k}, :);
        testLabels = labels(kIdx{k},:);

        cvResults = innerCrossValidation(type, trainData, trainLabels, 'Polynomial');
        [~, idx] = min([cvResults.Error]);
        result = cvResults(idx);

        if type
            model = fitcsvm(trainData, trainLabels, ...
                'KernelFunction', 'Polynomial', 'PolynomialOrder', result.q, ...
                'BoxConstraint', result.C);
        else 
            model = fitrsvm(trainData, trainLabels, ...
                'KernelFunction', 'Polynomial', 'Polynomial Order', result.q, ...
                'BoxConstraint', result.C, 'Epsilon', result.Epsilon);
        end

        results(k).C = result.C;
        results(k).q = result.q;
        if ~type
            results(k).Epsilon = result.Epsilon;
        end
        noOfSupportVectors =  height(model.SupportVectors);
        results(k).NoOfSupportVectors = noOfSupportVectors;
        results(k).PercentageOfSupportVectors = noOfSupportVectors/height(trainData);
        
        testPred = predict(model,testData);
        if type
            confusionMatrix = calculateConfusionMatrix(testLabels, testPred);
            testMetrics = struct('recall',NaN,'precision',NaN,'F1Score',NaN);
            testMetrics.recall = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FN);
            testMetrics.precision = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FP);
            testMetrics.F1Score = 2 * ( (testMetrics.precision*testMetrics.recall) / (testMetrics.precision+testMetrics.recall)  );
            results(k).metrics = testMetrics;
        else
            results(k).RMSE = calculateRMSE(testLabels, testPred);
        end
        results(k).labels = testPred.';
    end
end

