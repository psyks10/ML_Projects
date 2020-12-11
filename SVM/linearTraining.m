function results = linearTraining(type, features, labels)

    if type
        results = struct('Error', NaN,'NoOfSupportVectors', NaN);
    else
        results = struct('Epsilon', NaN, 'Error', NaN,'NoOfSupportVectors', NaN);
    end
    
    trainIdx = 1:ceil(height(features)*0.8);
    testIdx =  setdiff(1:height(features),trainIdx);

    trainData = features(trainIdx, :);
    trainTarg = labels(trainIdx,:);

    testData = features(testIdx, :);
    testTarg = labels(testIdx,:);
    
    if type
        epsilons = 1:1;
    else
        epsilons = {0.1};%,0.2,0.5,1];
    end
    
    for i = 1:length(epsilons)
        if type
            model = fitcsvm(trainData,trainTarg, 'KernelFunction','linear', 'BoxConstraint',1);
        else
            model = fitrsvm(trainData,trainTarg, 'KernelFunction','linear', 'BoxConstraint',1,'Epsilon', epsilons{i});
            results(i).Epsilon = epsilons{i};
        end
        results(i).Error = loss(model,testData, testTarg);
        noOfSupportVectors =  height(model.SupportVectors);
        results(i).NoOfSupportVectors = noOfSupportVectors;
        results(i).PercentageOfSupportVectors = noOfSupportVectors/height(trainData);
    end
end