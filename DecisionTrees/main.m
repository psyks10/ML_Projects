% Loading classification data
options = weboptions('ContentType','text');
breastCancer = webread('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data', options);
breastCancer = deblank(breastCancer);
breastCancer = splitlines(breastCancer);
breastCancer = regexp(breastCancer, ',+', 'split');
breastCancer = vertcat(breastCancer{:});
clear options

% Handle missing values
for feature = 1:width(breastCancer)
    indexes = strcmp(breastCancer(:,feature),'?');
    breastCancer(indexes,:) = []; 
end

% set up seed for random permutation (randperm) to use
seed = 101;
rng(seed);

% shuffle rows of cell array using seed
dataBC = breastCancer(randperm(size(breastCancer,1)),:);

% extract label-column from data set
labels = dataBC(:,1);

% extract feature-columns from data set
features = dataBC(:,2:10);

% create cell array of attribute IDs (feature names)
% attributeIDs = {'class', 'age', 'menopause', 'tumorSize', 'invNodes', 'nodeCaps', 'degMalig', 'breast', 'breastQuad', 'irradiat'};

% We need to now build the tree
decisionTree = decisionTreeLearning(features, labels);

