% Implementation of the ID3-algorithm to find the best attribute and
% threshold for a new node of the decision tree

function [bestAttribute, bestThreshold]  = chooseAttribute(features, labels, treetype)

    % Label is the first column in dataset
    dataset = horzcat(labels, features);

    totalExamples = height(labels);
    
    % Will hold the information gains for all attributes
    attributesInfGains = cell(width(features),1);
    
    % Intialise to -1 so we know if something has gone wrong
    bestInformationGain = -1;
    bestAttribute = -1;
    bestThreshold = -1;
    
    % Calculate the entropy for the dataset
    if treetype == 1
        positives = nnz(cell2mat(labels.label));
        positiveRatio = positives/totalExamples;
        negativeRatio = 1 - positiveRatio;
        datasetEntropy = calculateEntropy(positiveRatio, negativeRatio);
    else
        datasetEntropy = calculateStandardDeviation(labels);
    end
    
    % Iterate through attributes to calculate information gain values
    for currentAttribute = 1:width(features) 
                
        % Isolate selected attribute and get unique values
        uniqueValues = table2cell(unique(features(:,currentAttribute)));
       
        valuesEntropies = cell(length(uniqueValues),1);
        valuesInformationEntropies = cell(length(uniqueValues),1);
        
        % Calculate entropy for each unique value in the attribute
        for currentValue = 1:length(uniqueValues)

            % Get rows where currentAttribute == currentValue
            attrName = features.Properties.VariableNames{currentAttribute};
            [indexesToSubset,~] = find(strcmp(features.(attrName),uniqueValues{currentValue}));

            subset = dataset(indexesToSubset,:);
            
            if treetype == 1
                % All positives and negatives in the subsets
                positives = nnz(cell2mat(subset.label));
                negatives = height(subset) - positives;

                % Ratios
                positiveRatio = positives/height(subset);
                negativeRatio = 1 - positiveRatio;

                % Calculate entropy and information entropy for this value
                entropy = calculateEntropy(positiveRatio, negativeRatio);
                informationEntropy = (positives + negatives) / totalExamples * entropy;
                
            else
                % Positives and positive ratios
                positives = height(subset);
                positiveRatio = positives / height(features);
                
                entropy = calculateStandardDeviation(subset);
                informationEntropy = positiveRatio * entropy;
               
            end
            
            % Append to cell array to use later on
            valuesEntropies{currentValue} = entropy; 
            valuesInformationEntropies{currentValue} = informationEntropy;

        end
        
        % Calculate attribute information gain
        averageInformationEntropy = sum([valuesInformationEntropies{:}]);
        informationGain = datasetEntropy - averageInformationEntropy;

        % Update bestInformationGain, bestAttribute and bestThreshold
        if informationGain > bestInformationGain
            bestInformationGain = informationGain;
            bestAttribute = currentAttribute; % Column number
            [~,idx] = min([valuesInformationEntropies{:}]); % Get index of value with smallest information entropy
            bestThreshold = uniqueValues{idx}; % Set value as best threshold
        end 
        
        % Used to check if all the attributes have the same information
        % gain
        attributesInfGains{currentAttribute} = informationGain;
    end
    
    % If all the attributes have the same information gain, return a node
    if mean(cell2mat(attributesInfGains)) == bestInformationGain
        bestAttribute = -1;
    end
    
end