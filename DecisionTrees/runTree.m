function treePrediction = runTree(features, tree)
    % Traverse the tree and return the predictions
    
    treePrediction = cell(height(features),1);
    
    for index = 1:height(features)
        decisionTree = tree;
        while ~isempty(decisionTree.kids)
            iAttributeValue = string(features.(decisionTree.op){index});
            iThreshold = string(decisionTree.threshold);
            if strcmp(iAttributeValue, iThreshold)
               decisionTree = decisionTree.kids{1};
            else
                decisionTree = decisionTree.kids{2};
            end
        end
        treePrediction{index,1} = decisionTree.class;
    end
end