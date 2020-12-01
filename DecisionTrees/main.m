
%%%%%%%%%%%%%%%%%%%%%%%%%
%    Decision Trees     %
%       Team Matrix     %
%%%%%%%%%%%%%%%%%%%%%%%%%


% Please uncomment the first section to generate both trees using the whole
% dataset. Uncomment the second section to run K-Fold cross-validation.
% Please comment the unused section.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% First Section - Generate Decison Trees 
% This section loads each dataset and generates the decision trees 
% using the entire dataset

% Set up trees to generate
trees = {{1, 'Classification Tree'}, {0, 'Regression Tree'}};

for t = 1:length(trees)
    treeType = trees{t}{1};
    treeName = trees{t}{2};
    createTree(treeName, treeType);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Second Section - K-Fold
% This section splits the data in train, test and validation sets,
% runs K-Fold cross-validation and retuns evaluation metrics

% Classification
% [features, labels] = loadDataset(1);
% [trainEvaluationClass, testEvaluationClass] = runKfold(features, labels, 1);
% 
% Regression
% [features, labels] = loadDataset(0);
% [trainEvaluationReg, testEvaluationReg] = runKfold(features, labels, 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Functions

function createTree(treeName, treeType)
    [features, labels] = loadDataset(treeType);
    decisionTree = decisionTreeLearning(features, labels, treeType);
    DrawDecisionTree(decisionTree, treeName);
end


