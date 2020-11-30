function decisionTree = decisionTreeLearning(features, labels, treetype)
        
    decisionTree.op = "";
    decisionTree.kids = [];
    decisionTree.attribute = "";
    decisionTree.threshold = "";
    
    % If number of unique labels is 1 we return the only label
    %'unique labels'
    %height(unique(string(labels.label)))
    %'labels'
    %labels.label
    if height(unique(string(labels.label))) == 1
        decisionTree.class = mean(cell2mat(labels.label));
        return;
    end
    
    % Else get the attribute and threshold to use for splitting the node    
    [bestAttribute, bestThreshold] = chooseAttribute(features, labels, treetype);


    % If bestAttribute == -1 there was no bestAttribute to choose
    % so we use the majority. If there is no majority we use the most
    % common class in the dataset.
    if bestAttribute == -1
        if treetype == 1
            positives = nnz(cell2mat(labels.label));
            positivesRatio = positives/height(labels);
            if positivesRatio > 0.5
                decisionTree.class = 1;
            else
                decisionTree.class = 0;
            end
        else
            average = mean(cell2mat(labels.label));
            decisionTree.class = average;
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
    %[leftIndices,~] = find(strcmp(table2cell(features(:,bestAttribute)), bestThreshold));
    [leftIndices,~] = find(strcmp(features.(bestAttribute), bestThreshold));
    
    % Left kid subtree
    otherAttributes = setdiff(1:width(features), bestAttribute);
    leftKidFeatures = features(leftIndices,otherAttributes);
    leftKidLabels = labels(leftIndices,:);
    
    % Right kid subtree
    rightIndices = setdiff(1:height(features), leftIndices);
    rightKidFeatures = features(rightIndices,:);
    rightKidLabels = labels(rightIndices,:);
    
    % Generate kids subtrees
   if isempty(rightKidFeatures) || isempty(rightKidFeatures)
        decisionTree.kids = cell(1,1);
    else
        decisionTree.kids = cell(1,2);
    end

    decisionTree.kids{1} = decisionTreeLearning(leftKidFeatures, leftKidLabels, treetype);

    if ~isempty(rightKidFeatures) && ~isempty(rightKidLabels)
        decisionTree.kids{2} = decisionTreeLearning(rightKidFeatures, rightKidLabels, treetype);
    end
end