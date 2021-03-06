
% Function to load both regression and classifcation datasets
function [features, labels] = loadDataset(treeType)
    if treeType == 0
        [features, labels] = loadRegressionDataset();
    else
        [features, labels] = loadClassificationDataset();
    end
end
    
function [features, labels] = loadRegressionDataset()

    % Loading Regression data
    folder = 'Regression data';
    zipUrl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00544/ObesityDataSet_raw_and_data_sinthetic%20(2).zip';
    unzip(zipUrl, folder)
    file = dir([folder,'/*.csv']);
    obesityData = readtable([file.folder,'/',file.name]);
    rmdir(file.folder, 's');
    clear folder
    clear zipUrl
    clear file

    % Set up seed for random permutation (randperm) to use
    seed = 101;
    rng(seed);

    % Shuffle rows of cell array using seed
    randomIndices = randperm(size(obesityData,1));
    dataOD = obesityData(randomIndices,:);

    % Extract label-column from data set
    labels = dataOD(:,width(dataOD));
    labels.Properties.VariableNames = {'label'};
    
    % Encode label names
    labelNames = {'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', ...
        'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'};

    labelNamesMap = containers.Map(labelNames, 1:length(labelNames));

    for labelIdx = 1:length(labelNames)
        value = labelNames{labelIdx};
        indices = strcmp(labels.label, value);
        labels.label(indices) = {labelNamesMap(value)};
    end

    % Extract feature-columns from data set
    features = dataOD(:,1:width(dataOD)-1);

    numericalColumnsNearestInt = {'Age', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'};
    numericalColumnsTruncate = {'Height'};

    % Truncate height to generate bins of 10cm
    for colIdx = 1:length(numericalColumnsTruncate)
        col = numericalColumnsTruncate{colIdx};
        features.(col) = fix(features.(col)*10)/10;
        features.(col) = string(features.(col));
    end

    % Round to closest integer to generate bins
    for colIdx = 1:length(numericalColumnsNearestInt)
        col = numericalColumnsNearestInt{colIdx};
        features.(col) = round(features.(col));
        if numel(unique(string(features.(col)))) < 10
            bins = numel(unique(string(features.(col))));
        else
            bins = 10;
        end
        stepSize = ((max(features.(col)) - min(features.(col)) ) / bins);
        edges = min(features.(col)) : stepSize : max(features.(col));
        features.(col) = discretize(features.(col),edges);
        features.(col) = string(features.(col));
    end

    % Keep a map of the attribute names to obtain an attribute number when 
    % returning the nodes
    global attributeNames;
    colnames = features.Properties.VariableNames;
    attributeNames = containers.Map(colnames, 1:width(features));
end

function [features, labels] = loadClassificationDataset()  
    
    % Loading classification data
    options = weboptions('ContentType','text');
    breastCancer = webread('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data', options);
    breastCancer = deblank(breastCancer);
    breastCancer = splitlines(breastCancer);
    breastCancer = regexp(breastCancer, ',+', 'split');
    breastCancer = vertcat(breastCancer{:});

    % Handle missing values
    % Get rows in breastCancer with  values '?'
    [indices,~] = find(strcmp(breastCancer, '?')==1);
    % Remove the rows
    breastCancer(indices,:)=[];

    breastCancer = cell2table(breastCancer, 'VariableNames', {'class', 'age', 'menopause', 'tumorSize', 'invNodes', 'nodeCaps', 'degMalig', 'breast', 'breastQuad', 'irradiat'});
    clear options
    
    % set up seed for random permutation (randperm) to use
    seed = 101;
    rng(seed);

    % shuffle rows of cell array using seed
    randomIndices = randperm(size(breastCancer,1));
    dataBC = breastCancer(randomIndices,:);

    % Extract label-column from data set
    labels = dataBC(:,1);
    labels.Properties.VariableNames = {'label'}; 

    % Encode label names to binary
    labelNames = {'no-recurrence-events', 'recurrence-events'};
    
    labelNamesMap = containers.Map(labelNames, 0:length(labelNames)-1);

    for labelIdx = 1:length(labelNames)
        value = labelNames{labelIdx};
        indices = strcmp(labels.label, value);
        labels.label(indices) = {labelNamesMap(value)};
    end

    % Extract feature-columns from data set
    features = dataBC(:,2:width(dataBC));
    
    % Keep a map of the attribute names to obtain an attribute number when 
    % returning the nodes
    global attributeNames;
    attributeNames = containers.Map({'age', 'menopause', 'tumorSize', 'invNodes', 'nodeCaps', 'degMalig', 'breast', 'breastQuad', 'irradiat'}, {1,2,3,4,5,6,7,8,9});

end
