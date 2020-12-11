function results = innerCrossValidation(type, features, labels, kernelFunction)    
    
    if strcmp(kernelFunction,'Polynomial')
        kernelParameterName = 'PolynomialOrder';
        kernelParameter = 'q';
        options = 1:2;
    else
        kernelParameterName = 'KernelScale';
        kernelParameter = 'Gamma';
        options = 2.^(-5:2:15);
    end
    
    if type
        results = struct('C', NaN, kernelParameter, NaN, 'Error', NaN,'NoOfSupportVectors', NaN);
    else
        results = struct('C', NaN, kernelParameter, NaN, 'Epsilon', NaN, 'Error', NaN,'NoOfSupportVectors', NaN);
    end
    
    gridC = 2.^(-5:2:15);
   
    if type
        epsilons = 1:1;
    else
        epsilons = 0:0.1:1;
    end

    innerkIdx = getkFoldIdx(height(labels)).';

    % Grid Search
    for C = gridC
        for kernelParameterValue = options
            for epsilon = epsilons
                foldscores = cell(10,2);
                % inner cross validation
                for innerk = 1:10
                    innertrainIdx = horzcat(innerkIdx{setdiff(1:10,innerk)}).';

                    innertrainData = features(innertrainIdx, :);
                    innertrainLabels = labels(innertrainIdx,:);

                    innertestData = features(innerkIdx{innerk}, :);
                    innertestLabels = labels(innerkIdx{innerk},:);

                    if type
                        cvSVMModel = fitcsvm(innertrainData, innertrainLabels, ...
                            'KernelFunction', kernelFunction, kernelParameterName, kernelParameterValue, ...
                            'BoxConstraint', C);
                        
                    else
                        cvSVMModel = fitrsvm(innertrainData, innertrainLabels, ...
                            'KernelFunction', kernelFunction, kernelParameterName, kernelParameterValue, ...
                            'BoxConstraint', C,'Epsilon', epsilon);
                        
                    end

                    innertestPred = predict(cvSVMModel, innertestData);
                    
                    if type
                        confusionMatrix = calculateConfusionMatrix(innertestLabels, innertestPred);
                        recall = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FN);
                        precision = confusionMatrix.TP / (confusionMatrix.TP+confusionMatrix.FP);
                        F1Score = 2 * ( (precision*recall) / (precision+recall));
                        foldscores{innerk,1} = F1Score;
                    else
                        foldscores{innerk,1} = calculateRMSE(innertestLabels, innertestPred);
                    end
                    foldscores{innerk,2} = height(cvSVMModel.SupportVectors);
                end

                avgScore = mean(cell2mat(foldscores(:,1)));
                n = length(results)+1;
                results(n).C = C;
                results(n).(kernelParameter) = kernelParameterValue;
                if ~type
                    results(n).Epsilon = epsilon;
                end
                results(n).Error = avgScore;
                noOfSupportVectors = mean(cell2mat(foldscores(:,2)));
                results(n).NoOfSupportVectors = noOfSupportVectors;
                results(n).PercentageOfSupportVectors = noOfSupportVectors/height(innertrainData);
            end
        end
    end
end


