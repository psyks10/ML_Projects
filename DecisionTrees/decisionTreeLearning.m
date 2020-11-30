function decisionTree = decisionTreeLearning(features, labels)
       
    decisionTree.op = "";
    decisionTree.kids = [];
    decisionTree.attribute = "";
    decisionTree.threshold = "";
    
    % If number of unique labels is 1 we return the only label
    if height(unique(labels.class)) == 1
        prediction = unique(string(labels.class));
        if string(prediction) == 'no-recurrence-events'
            decisionTree.class = 0;
        elseif string(prediction) == 'recurrence-events'
            decisionTree.class = 1;
        end
        return;
    end
    
    % Else get the attribute and threshold to use for splitting the node
    % If bestAttribute == -1 there was no bestAttribute to choose
    % so we use the majority. If there is no majority we use the most
    % common class in the dataset.
    [bestAttribute, bestThreshold] = chooseAttribute(features, labels);
    if bestAttribute == -1
        positives = nnz(strcmp(labels.class, 'recurrence-events'));
        negatives = height(labels) - positives;
        if positives > negatives
            decisionTree.class = 1;
        else
            decisionTree.class = 0;
        end
        return;
    end
    
    decisionTree.op = features.Properties.VariableNames{bestAttribute};
    % Assign the attribute and threshold-values from the ID3 algorithm chooseAttribute() to the relevant fields of the tree-struct
    % set attribute number using name in map
    global attributeNames
    decisionTree.attribute = attributeNames(decisionTree.op);
    decisionTree.threshold = bestThreshold;
    
    % Get indexes of rows where attribute values are equal to the threshold
    [leftIndices,~] = find(strcmp(table2cell(features(:,bestAttribute)), bestThreshold));
    
    % Left kid subtree
    otherAttributes = setdiff(1:width(features), bestAttribute);
    leftKidFeatures = features(leftIndices,otherAttributes);
    leftKidLabels = labels(leftIndices,:);
    
    % Right kid subtree
    rightIndices = setdiff(1:height(features), leftIndices);
    rightKidFeatures = features(rightIndices,:);
    rightKidLabels = labels(rightIndices,:);
    
    % Generate kids subtrees
    decisionTree.kids = cell(1,2);
    decisionTree.kids{1} = decisionTreeLearning(leftKidFeatures, leftKidLabels);
    decisionTree.kids{2} = decisionTreeLearning(rightKidFeatures, rightKidLabels);
    
end