% Loading classification data
options = weboptions('ContentType','text');
breastCancer = webread('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data', options);
breastCancer = deblank(breastCancer);
breastCancer = splitlines(breastCancer);
breastCancer = regexp(breastCancer, ',+', 'split');
breastCancer = vertcat(breastCancer{:});
clear options


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

decisionTree = decisionTreeLearning(features, labels);

function decisionTree = decisionTreeLearning(features, labels)

    % Initialise the tree
    decisionTree = struct('op', [], 'kids', [], 'prediction', [], 'attribute', [], 'threshold', []);
    
    % If number of unique labels is 1, then return the label
    % (unique() only works with strings, so convert label from cell array to string array)
    %if numel(unique(string(labels))) == 1
        %decisionTree(end+1) = struct('op', '', 'kids', [], 'prediction', unique(string(labels)), 'attribute', '', 'threshold', '');
        %return;
    %end
    
    % If entropy is not 0, split data set depending on best attribute
    
    % Get the attribute and threshold for the new node
    [bestAttribute, bestThreshold] = chooseAttribute(features, labels);
    
    % Assign the attribute and threshold-values from the ID3 algorithm
    % chooseAttribute() to the relevant fields of the tree-struct
    decisionTree.attribute = bestAttribute;
    decisionTree.threshold = bestThreshold;
    % Set other fields of the tree-struct!
    
end

% Implementation of the ID3-algorithm to find the best attribute and
% threshold for a new node of the decision tree
function [iAttribute, iThreshold]  = chooseAttribute(features, labels)
    tot = length(labels);
    informationGain = 0;
    iAttribute = 1;
    % Iterate through attributes to calculate information gain values
    for n = 1:9 % number of features
        % positive occurrences of selected attribute in data set
        iAttribute = features(1:tot, 5:5); % example for nodeCaps, change 5 to n
        iAttributePos = nnz(strcmp(iAttribute, 'yes'));
        ratio = iAttributePos/tot; % rounds division
        entropy = calculateEntropy(ratio);
        % if entropy is 0, then move to next iteration!
        if entropy ~= 0
            iInformationGain = calculateInformationGain(entropy, ratio);

            % compare information gain: if increased, update iAttribute
            if iInformationGain > informationGain
                informationGain = iInformationGain;
                iAttribute = n;
            end
        end
    end
    iThreshold = 0;
end

function entropy = calculateEntropy(ratio)
    entropy = - ratio * log2(ratio);
end

function informationGain = calculateInformationGain(entropy, ratio)
    informationGain = entropy - ratio * entropy;
end