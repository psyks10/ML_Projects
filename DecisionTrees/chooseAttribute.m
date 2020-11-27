% Implementation of the ID3-algorithm to find the best attribute and
% threshold for a new node of the decision tree

function [bestAttribute, bestThreshold]  = chooseAttribute(features, labels)

    % Label is the first column in dataset
    dataset = horzcat(labels, features);
    uniqueLabels = unique(labels.class);

    totalExamples = height(labels);
    
    % Intialise to -1 so we know if something has gone wrong
    bestInformationGain = -1;
    bestAttribute = -1;
    bestThreshold = -1; 
    
    positives = nnz(strcmp(labels, uniqueLabels{1}));
    positiveRatio = positives/totalExamples;
    negativeRatio = 1 - positiveRatio;
    datasetEntropy = calculateEntropy(positiveRatio, negativeRatio);
    
    width(features) 
    % Iterate through attributes to calculate information gain values
    for currentAttribute = 1:width(features) 
        
        currentAttribute
        
        % Isolate selected attribute and get unique values
        % attribute = features(:,feature); 
        uniqueValues = table2cell(unique(features(:,currentAttribute)));
        uniqueValues
       
        valuesEntropies = {length(uniqueValues),1};
        valuesInformationEntropies = {length(uniqueValues),1};
        
        % Calculate entropy and information entropy for each value
        for currentValue = 1:length(uniqueValues)
            
            % Get rows where currentAttribute == currentValue
            % This is not working
            indexesToSubset = strcmp(features(:,currentAttribute),string(currentValue));
            subset = dataset(indexesToSubset,:);
            
            % All positives and negatives in the subsets
            positives = nnz(strcmp(subset(:,1), uniqueLabels{1}));
            negatives = height(subset) - positives;
            
            % Ratios
            positiveRatio = positives/height(subset);
            negativeRatio = 1 - positiveRatio;
            
            positiveRatio
            negativeRatio
            
            % Calculate entropy and information entropy for this value
            entropy = calculateEntropy(positiveRatio, negativeRatio);
            informationEntropy = (positives + negatives) / totalExamples * entropy;
            
            % Append to cell array to use later on
            valuesEntropies{currentValue} = entropy; 
            valuesInformationEntropies{currentValue} = informationEntropy;
        end
        
        % Calculate attribute information gain
        averageInformationEntropy = sum([valuesInformationEntropies{:}]);
        informationGain = datasetEntropy - averageInformationEntropy;
        
        if informationGain > bestInformationGain
            bestInformationGain = informationGain;
            bestAttribute = currentAttribute; % Column number
            [~,idx] = min([valuesInformationEntropies{:}]); % Get best threshold index 
            bestThreshold = uniqueValues{idx}; % Get value for best threshold
        end
        
    end
    bestAttribute
end

