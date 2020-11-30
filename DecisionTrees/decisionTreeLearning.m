function decisionTree = decisionTreeLearning(features, labels)
       
    global counter;
    counter = counter+1;
    
    % If number of unique labels is 1 then everything has the same label, we return the only label
    if height(unique(labels.class)) == 1
        prediction = unique(string(labels.class));
        if string(prediction) == 'no-recurrence-events'
            decisionTree.class = 0;
        elseif string(prediction) == 'recurrence-events'
            decisionTree.class = 1;
        else
            decisionTree.class = 3;
        end
        decisionTree.op = "";
        decisionTree.kids = [];
        decisionTree.attribute = "";
        decisionTree.threshold = "";
        return;
    end
    
    % Initialise the tree
    decisionTree = struct('op', [], 'kids', [], 'class', [], 'attribute', [], 'threshold', []);

    % Else get the attribute and threshold to use for splitting the node
    [bestAttribute, bestThreshold] = chooseAttribute(features, labels);
    if bestAttribute == -1
        positives = nnz(strcmp(labels.class, 'recurrence-events'));
        negatives = height(labels) - positives;
        if positives > negatives
            decisionTree.class = 1;
        else
            decisionTree.class = 0;
        end
        decisionTree.op = "";
        decisionTree.kids = [];
        decisionTree.attribute = "";
        decisionTree.threshold = "";
        return;
    end
    
    decisionTree.op = features.Properties.VariableNames{bestAttribute};
    % Assign the attribute and threshold-values from the ID3 algorithm chooseAttribute() to the relevant fields of the tree-struct
    % set attribute number using name in map
    global attributeNames
    decisionTree.attribute = attributeNames(decisionTree.op);
    decisionTree.threshold = bestThreshold;
    
    % get indexes of rows where attribute values meet the threshold
    [leftIndices,~] = find(strcmp(table2cell(features(:,bestAttribute)), bestThreshold));
    
    otherAttributes = setdiff(1:width(features), bestAttribute);
    leftKidFeatures = features(leftIndices,otherAttributes);
    leftKidLabels = labels(leftIndices,:);
    
    % delete rows at indexesToSubset
    rightIndices = setdiff(1:height(features), leftIndices);
    rightKidFeatures = features(rightIndices,:);
    rightKidLabels = labels(rightIndices,:);
    
    % only set kids if there are rows in the feature tables
    decisionTree.kids = cell(1,2);
    decisionTree.kids{1} = decisionTreeLearning(leftKidFeatures, leftKidLabels);
    decisionTree.kids{2} = decisionTreeLearning(rightKidFeatures, rightKidLabels);
    
end