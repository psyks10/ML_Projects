function decisionTree = decisionTreeLearning(features, labels)
       
    global counter;
    counter = counter+1;
    counter
    
    % Initialise the tree
    decisionTree = struct('op', [], 'kids', [], 'prediction', [], 'attribute', [], 'threshold', []);
    
    % If number of unique labels is 1 then everything has the same label, we
    % return the only label
    if numel(unique(string(labels.class))) == 1
        decisionTree(end+1) = struct('op', '', 'kids', [], 'prediction', unique(string(labels)), 'attribute', '', 'threshold', '');
        return;
    end

    % Else wet the attribute and threshold to use for splitting the node
    [bestAttribute, bestThreshold] = chooseAttribute(features, labels);
    
    % Assign the attribute and threshold-values from the ID3 algorithm
    % chooseAttribute() to the relevant fields of the tree-struct
    decisionTree.attribute = bestAttribute;
    decisionTree.threshold = bestThreshold;
    
    decisionTree.op = features.Properties.VariableNames{bestAttribute};
    
    indexesToSubset = strcmp(features(:,bestAttribute),string(bestThreshold));
    otherAttributes = setdiff(1:width(features), bestAttributes);
    leftKidFeatures = features(indexesToSubset,otherAttributes);
    leftKidLabels = labels(indexesToSubset,:);
    indexesToSubset = ~indexesToSubset;
    rightKidFeatures = features(indexesToSubset,:);
    rightKidLabels = labels(indexesToSubset,:);
    decisionTree.kids = cell(2,1);
    decisionTree.kids{1} = decisionTreeLearning(leftKidFeatures, leftKidLabels);
    decisionTree.kids{2} = decisionTreeLearning(rightKidFeatures, rightKidLabels);
    
    % Set other fields of the tree-struct!
    
end