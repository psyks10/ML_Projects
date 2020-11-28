% Loading classification data
options = weboptions('ContentType','text');
breastCancer = webread('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data', options);
breastCancer = deblank(breastCancer);
breastCancer = splitlines(breastCancer);
breastCancer = regexp(breastCancer, ',+', 'split');
breastCancer = vertcat(breastCancer{:});
% Convert all columns to categorical

% Handle missing values
% Get rows in breastCancer with  values '?'
[indices,~] = find(strcmp(breastCancer, '?')==1);
% Remove the rows
breastCancer(indices,:)=[];

breastCancer = cell2table(breastCancer, 'VariableNames', {'class', 'age', 'menopause', 'tumorSize', 'invNodes', 'nodeCaps', 'degMalig', 'breast', 'breastQuad', 'irradiat'});
breastCancer.Properties.VariableNames = {'class', 'age', 'menopause', 'tumorSize', 'invNodes', 'nodeCaps', 'degMalig', 'breast', 'breastQuad', 'irradiat'};
clear options

% Counter
global counter;
counter = 0;

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
global attributeNames;
attributeNames = containers.Map({'age', 'menopause', 'tumorSize', 'invNodes', 'nodeCaps', 'degMalig', 'breast', 'breastQuad', 'irradiat'}, {1,2,3,4,5,6,7,8,9});

% We need to now build the tree
decisionTree = decisionTreeLearning(features, labels);

