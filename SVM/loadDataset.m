
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

    % set up seed for random permutation (randperm) to use
    seed = 101;
    rng(seed);

    % shuffle rows of cell array using seed
    dataOD = obesityData(randperm(size(obesityData,1)),:);

    % extract label-column from data set
    labels = dataOD(:,width(dataOD));
    labels.Properties.VariableNames = {'label'};
    labelNames = {'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', ...
        'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'};

    labelNamesMap = containers.Map(labelNames, 1:length(labelNames));

    for labelIdx = 1:length(labelNames)
        value = labelNames{labelIdx};
        indices = strcmp(labels.label, value);
        labels.label(indices) = {labelNamesMap(value)};
    end

    % extract feature-columns from data set
    features = dataOD(:,1:width(dataOD)-1);

    % Keep a map of the attribute names to obtain an attribute number when 
    % returning the nodes
    global attributeNames;
    colnames = features.Properties.VariableNames;
    attributeNames = containers.Map(colnames, 1:width(features));
    labels = cell2mat(table2cell(labels(:,:)));

end

function [features, labels] = loadClassificationDataset()  
    
    % Loading classification data
    websave('breastCancerData.csv', 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data');
    breastCancer = readtable('breastCancerData.csv');
    breastCancer.Properties.VariableNames = {'class', 'age', 'menopause', 'tumorSize', 'invNodes', 'nodeCaps', 'degMalig', 'breast', 'breastQuad', 'irradiat'};
    delete 'breastCancerData.csv';

    % Handle missing values
    % Get rows in breastCancer with  values '?'
    [indices, ~] = find(strcmp(table2cell(breastCancer), '?'));
    % Remove the rows
    breastCancer(indices,:)=[];
    
    % Shuffle rows of cell array using seed
    dataBC = breastCancer(randperm(size(breastCancer,1)),:);

    % Extract label-column from data set
    labels = dataBC(:,1);
    labels.Properties.VariableNames = {'label'}; 

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
    
    labels = cell2mat(table2cell(labels(:,:)));
end
