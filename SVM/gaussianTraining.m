function results = gaussianTraining(type, features, labels)    


    if type
        results = struct('C', NaN, 'Gamma', NaN,'NoOfSupportVectors', NaN, 'labels', NaN);
    else
        results = struct('C', NaN, 'Gamma', NaN, 'Epsilon', NaN, 'RMSE', NaN,'NoOfSupportVectors', NaN, 'labels', NaN);
    end
    
    kIdx = getkFoldIdx(ceil(height(labels)));
    

        
    for k = 1:10
      
        trainIdx = horzcat(kIdx{setdiff(1:10,k)}).';
        trainData = features(trainIdx, :);
        trainLabels = labels(trainIdx,:);
        testData = features(kIdx{k}, :);
        testTarg = labels(kIdx{k},:);
        
        cvResults = innerCrossValidation(type, features, labels, 'gaussian');
        [~, idx] = min([cvResults.Error]);
        result = cvResults(idx);
        
        if type
            model = fitcsvm(trainData, trainLabels, ...
                'KernelFunction', 'RBF', 'KernelScale', result.Gamma, ...
                'BoxConstraint', result.C);
        else 
            model = fitrsvm(trainData, trainLabels, ...
                'KernelFunction', 'RBF', 'KernelScale', result.Gamma, ...
                'BoxConstraint', result.C, 'Epsilon', result.Epsilon);
        end

        results(k).C = result.C;
        results(k).Gamma = result.Gamma;
        if ~type
            results(k).Epsilon = result.Epsilon;
        end
        noOfSupportVectors =  height(model.SupportVectors);
        results(k).NoOfSupportVectors = noOfSupportVectors;
        results(k).PercentageOfSupportVectors = noOfSupportVectors/height(trainData);
         
        testPred = predict(model,testData);
        if type
            confusionMatrix = calculateConfusionMatrix(testTarg, testPred);
            testMetrics = struct('recall',NaN,'precision',NaN,'F1Score',NaN);
            testMetrics.recall = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FN);
            testMetrics.precision = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FP);
            testMetrics.F1Score = 2 * ( (testMetrics.precision*testMetrics.recall) / (testMetrics.precision+testMetrics.recall)  );
            results(k).metrics = testMetrics;
        else
            results(k).RMSE = calculateRMSE(testTarg, testPred);
        end
        results(k).labels = testPred.';
    end
end

