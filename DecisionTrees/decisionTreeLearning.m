function decisionTree = decisionTreeLearning(features, labels)
       
    global counter;
    counter = counter+1;
    
    
    % If number of unique labels is 1 then everything has the same label, we
    % return the only label
    if numel(unique(string(labels.class))) == 1
        decisionTree = struct('op', '', 'kids', [], 'prediction', unique(string(labels.class)), 'attribute', '', 'threshold', '');
        return;
    end
    
    % Initialise the tree
    decisionTree = struct('op', [], 'kids', [], 'prediction', [], 'attribute', [], 'threshold', []);

    % Else wet the attribute and threshold to use for splitting the node
    [bestAttribute, bestThreshold] = chooseAttribute(features, labels);
    
    % if there is no best attribute return empty decisionTree
    if bestAttribute==-1
        decisionTree = [];
        return
    end
    
    decisionTree.op = features.Properties.VariableNames{bestAttribute};
    % Assign the attribute and threshold-values from the ID3 algorithm
    % chooseAttribute() to the relevant fields of the tree-struct
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
    rightKidFeatures = features(~indexesToSubset,:);
    rightKidLabels = labels(~indexesToSubset,:);
    
%     rightKidFeatures = features;
%     rightKidFeatures(indexesToSubset,:)=[];
%     rightKidLabels = labels;
%     rightKidLabels(indexesToSubset,:)=[];
    
    decisionTree.kids = cell(1,2);
    % only set kids if there are rows in the feature tables
    if height(leftKidFeatures)~=0
        decisionTree.kids{1} = decisionTreeLearning(leftKidFeatures, leftKidLabels); 
    end
    if height(rightKidFeatures)~=0
        decisionTree.kids{2} = decisionTreeLearning(rightKidFeatures, rightKidLabels);
    end
    
    % Set other fields of the tree-struct!
    
end