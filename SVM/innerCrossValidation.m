function results = innerCrossValidation(type, features, labels, kernelFunction)    
    typeBool = type==1;
    
    if strcmp(kernelFunction,'Polynomial')
        kernelParameterName = 'Polynomial Order';
        kernelParameter = 'q';
        options = 1:10;
    else
        kernelParameterName = 'KernelScale';
        kernelParameter = 'Gamma';
        options = 2.^(-5:2:15);
    end
    
    if typeBool
        results = struct('C', NaN, kernelParameter, NaN, 'Error', NaN,'NoOfSupportVectors', NaN);
    else
        results = struct('C', NaN, kernelParameter, NaN, 'Epsilon', NaN, 'Error', NaN,'NoOfSupportVectors', NaN);
    end
    
    gridC = 2.^(-5:2:15);
   
    if typeBool
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
                    innertrainTarg = labels(innertrainIdx,:);

                    innertestData = features(innerkIdx{innerk}, :);
                    innertestTarg = labels(innerkIdx{innerk},:);

                    if typeBool
                        cvSVMModel = fitcsvm(innertrainData, innertrainTarg, ...
                            'KernelFunction', kernelFunction, kernelParameterName, kernelParameterValue, ...
                            'BoxConstraint', C);
                    else
                        cvSVMModel = fitrsvm(innertrainData, innertrainTarg, ...
                            'KernelFunction', kernelFunction, kernelParameterName, kernelParameterValue, ...
                            'BoxConstraint', C,'Epsilon', epsilon);
                    end

                    foldscores{innerk,1} = loss(cvSVMModel,innertestData, innertestTarg);
                    foldscores{innerk,2} = height(cvSVMModel.SupportVectors);
                end

                avgScore = mean(cell2mat(foldscores(:,1)));
                n = length(results)+1;
                results(n).C = C;
                results(n).(kernelParameter) = kernelParameterValue;
                if ~typeBool
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


