
%%%%%%%%%%%%%%%%%%%%%%%%%
%    Decision Trees     %
%       Team Matrix     %
%%%%%%%%%%%%%%%%%%%%%%%%%

% Set up seed 
seed = 101;
rng(seed);

% Set up trees to generate
trees = {{1, 'Classification Tree'}, {0, 'Regression Tree'}};

% Generate decison trees
for t = 1:length(trees)
    treeType = trees{t}{1};
    treeName = trees{t}{2};
    createTree(treeName, treeType);
end

% K-Fold

% Split data 80/20
[features, labels] = loadDataset(1);
accuracyClass = runKfold(features, labels, 1);

[features, labels] = loadDataset(0);
accuracyReg = runKfold(features, labels, 0);

function createTree(treeName, treeType)
    [features, labels] = loadDataset(treeType);
    decisionTree = decisionTreeLearning(features, labels, treeType);
    DrawDecisionTree(decisionTree, treeName);
end


