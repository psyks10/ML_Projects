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
        
        if type
            cvResults = innerCrossValidation(type, trainData, trainLabels, 'gaussian');
        else
            subsets = 1:ceil(height(trainData)*0.05);
            cvResults = innerCrossValidation(type, trainData(subsets,:), trainLabels(subsets,:), 'gaussian');
        end
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
            results(k).metrics = testMetrics;

        else
            results(k).RMSE = calculateRMSE(testTarg, testPred);
        end
        results(k).labels = testPred.';
    end
end

