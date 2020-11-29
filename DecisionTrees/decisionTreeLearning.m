function decisionTree = decisionTreeLearning(features, labels, class)
       
    global counter;
    counter = counter+1;
    
    % If number of unique labels is 1 then everything has the same label, we
    % return the only label
    %decisionTree = struct('op', '', 'kids', cell(1,2), 'prediction', unique(string(labels.class)), 'attribute', '', 'threshold', '');
    if numel(unique(string(labels.class))) == 1 || isempty(features)
        decisionTree.prediction = unique(string(labels.class));
        decisionTree.op = "";
        decisionTree.kids = [];
        decisionTree.attribute = "";
        decisionTree.threshold = "";
        decisionTree.class = class;
        return;
    end
    
    % Initialise the tree
    decisionTree = struct('op', [], 'kids', [], 'prediction', [], 'attribute', [], 'threshold', [], 'class', class);

    % Else get the attribute and threshold to use for splitting the node
    [bestAttribute, bestThreshold] = chooseAttribute(features, labels);
    
    decisionTree.op = features.Properties.VariableNames{bestAttribute};
    % Assign the attribute and threshold-values from the ID3 algorithm chooseAttribute() to the relevant fields of the tree-struct
    % set attribute number using name in map
    global attributeNames
    decisionTree.attribute = attributeNames(decisionTree.op);
    decisionTree.threshold = bestThreshold;
    
    % get indexes of rows where attribute values meet the threshold
    [indexesToSubset,~] = find(strcmp(table2cell(features(:,bestAttribute)), bestThreshold));
    
    otherAttributes = setdiff(1:width(features), bestAttribute);
    leftKidFeatures = features(indexesToSubset,otherAttributes);
    leftKidLabels = labels(indexesToSubset,:);
    
    % delete rows at indexesToSubset
    rightKidFeatures = features;
    rightKidFeatures(indexesToSubset,:)=[];
    rightKidLabels = labels;
    rightKidLabels(indexesToSubset,:)=[];
    
    % only set kids if there are rows in the feature tables
    decisionTree.kids = cell(1,2);
    decisionTree.kids{1} = decisionTreeLearning(leftKidFeatures, leftKidLabels, class);
    decisionTree.kids{2} = decisionTreeLearning(rightKidFeatures, rightKidLabels, class);
    
end