% Implementation of the ID3-algorithm to find the best attribute and
% threshold for a new node of the decision tree

function [bestAttribute, bestThreshold]  = chooseAttribute(features, labels, treetype)

    % Label is the first column in dataset
    dataset = horzcat(labels, features);
    % uniqueLabels = unique(labels.label);

    totalExamples = height(labels);
    
    % Intialise to -1 so we know if something has gone wrong
    bestInformationGain = -1;
    bestAttribute = -1;
    bestThreshold = -1;
    
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
        
        currentAttribute
                
        % Isolate selected attribute and get unique values
        % attribute = features(:,feature); 
        uniqueValues = table2cell(unique(features(:,currentAttribute)));
       
        valuesEntropies = {length(uniqueValues),1};
        valuesInformationEntropies = {length(uniqueValues),1};
        
        % Calculate entropy and information entropy for each value
        for currentValue = 1:length(uniqueValues)
            'Value checked'
            currentValue
            'UniqueValues 1'
            uniqueValues
            % Get rows where currentAttribute == currentValue
            attrName = features.Properties.VariableNames{currentAttribute};
            % [indexesToSubset,~] = find(strcmp(table2cell(features(:,currentAttribute)),uniqueValues{currentValue}));
            [indexesToSubset,~] = find(strcmp(features.(attrName),uniqueValues{currentValue}));
            if isempty(indexesToSubset)
                indexesToSubset
            end
            subset = dataset(indexesToSubset,:);
            indexesToSubset
            dataset
            
            if treetype == 1
                'treetype == 1'
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
                'treetype == 0'
                positives = height(subset);
                positiveRatio = positives / height(features);
                
                'Second call'
                entropy = calculateStandardDeviation(subset);
                informationEntropy = positiveRatio * entropy;
               
            end
            
            % Append to cell array to use later on
            'Appending'
            valuesEntropies{currentValue} = entropy; 
            valuesInformationEntropies{currentValue} = informationEntropy;
            'valuesInformationEntropies'
            valuesInformationEntropies

        end
        
        % Calculate attribute information gain
        averageInformationEntropy = sum([valuesInformationEntropies{:}]);
        informationGain = datasetEntropy - averageInformationEntropy;


        % Update bestInformationGain, bestAttribute and bestThreshold
        if informationGain > bestInformationGain
            bestInformationGain = informationGain;
            bestAttribute = currentAttribute; % Column number
            [~,idx] = min([valuesInformationEntropies{:}]); % Get index of value with smallest information entropy
            idx
            'bestAttribute'
            bestAttribute
            'UniqueValues 2'
            uniqueValues
            valuesInformationEntropies
            bestThreshold = uniqueValues{idx}; % Set value as best threshold
        end     
    end
end